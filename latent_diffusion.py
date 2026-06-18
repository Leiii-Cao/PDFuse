"""
This module includes LDM-based inverse problem solvers.
Forward operators follow DPS and DDRM/DDNM.
"""

import time
from typing import Callable, Dict, Optional

import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, StableDiffusionPipeline
from tqdm import tqdm

from utils.LIME import LIME_PyTorch
from utils.degration import (
    build_degradation_pipeline,
    strip_exposure_operators,
)
from utils.exposure_weight import exposure_fusion
from utils.utils import (
    gaussian_blur,
    image_gradients,
    make_laplacian_kernel,
    make_sobel_kernels,
)


__SOLVER__ = {}
DEFAULT_LIGHT_ITERATIONS = 10
DEFAULT_LIGHT_STRATEGY = 2


def build_pixel_target(
    image_a: torch.Tensor,
    image_b: torch.Tensor,
    fusion_task: str,
) -> torch.Tensor:
    """Build the pixel supervision target for different fusion tasks."""
    if fusion_task in ("IVF", "MMIF"):
        return torch.max(image_a, image_b)
    if fusion_task == "MFIF":
        return (image_a + image_b) / 2
    if fusion_task == "MEIF":
        mean_a = torch.mean(image_a)
        mean_b = torch.mean(image_b)
        dist_a = torch.abs(mean_a - 0.5)
        dist_b = torch.abs(mean_b - 0.5)
        score_a = 1.0 / (dist_a + 1e-8)
        score_b = 1.0 / (dist_b + 1e-8)
        weight_a = score_a / (score_a + score_b)
        weight_b = score_b / (score_a + score_b)
        target = weight_a * image_a + weight_b * image_b
        gain = 0.5 / (torch.mean(target) + 1e-8)
        gain = torch.clamp(gain, 0.5, 2.0)
        return (target * gain).clamp(0, 1)
    raise ValueError(
        "fusion_task must be one of: IVF, MMIF, MFIF, MEIF"
    )


def build_pixel_target_and_weights(
    image_a: torch.Tensor,
    image_b: torch.Tensor,
    fusion_task: str,
):
    if fusion_task in ("IVF", "MMIF"):
        weight_a = (image_a >= image_b).to(image_a.dtype)
        weight_b = 1.0 - weight_a
        return torch.max(image_a, image_b), weight_a, weight_b
    if fusion_task == "MFIF":
        weight_a = torch.full_like(image_a, 0.5)
        weight_b = torch.full_like(image_b, 0.5)
        return (image_a + image_b) / 2, weight_a, weight_b
    if fusion_task == "MEIF":
        mean_a = torch.mean(image_a)
        mean_b = torch.mean(image_b)
        score_a = 1.0 / (torch.abs(mean_a - 0.5) + 1e-8)
        score_b = 1.0 / (torch.abs(mean_b - 0.5) + 1e-8)
        scalar_a = score_a / (score_a + score_b)
        scalar_b = score_b / (score_a + score_b)
        target = scalar_a * image_a + scalar_b * image_b
        gain = torch.clamp(0.5 / (torch.mean(target) + 1e-8), 0.5, 2.0)
        return (target * gain).clamp(0, 1), torch.ones_like(image_a) * scalar_a, torch.ones_like(image_b) * scalar_b
    raise ValueError("fusion_task must be one of: IVF, MMIF, MFIF, MEIF")


def Get_Light_params(
    measurement,
    use_lowlight=True,
    use_overexposure=True,
    iterations=DEFAULT_LIGHT_ITERATIONS,
    strategy=DEFAULT_LIGHT_STRATEGY,
):
    enhancer = LIME_PyTorch(iterations=iterations, strategy=strategy)
    padded = F.pad(measurement, pad=(64, 64, 64, 64), mode="reflect")
    measurement_under = measurement
    measurement_over = measurement

    if use_lowlight:
        enhancer.load(padded.squeeze(0).permute(1, 2, 0), device=measurement.device)
        measurement_under, _ = enhancer.enhance()
        measurement_under = measurement_under[:, :, 64 : 64 + measurement.shape[2], 64 : 64 + measurement.shape[3]]

    if use_overexposure:
        enhancer.load((1 - padded).squeeze(0).permute(1, 2, 0), device=measurement.device)
        measurement_over, _ = enhancer.enhance()
        measurement_over = measurement_over[:, :, 64 : 64 + measurement.shape[2], 64 : 64 + measurement.shape[3]]
        measurement_over = 1 - measurement_over

    if use_lowlight and use_overexposure:
        measurement_consist = exposure_fusion(
            torch.cat((measurement_under, measurement, measurement_over), dim=0)
        ).clamp(0, 1).unsqueeze(0)
    elif use_lowlight:
        measurement_consist = measurement_under
    elif use_overexposure:
        measurement_consist = measurement_over
    else:
        measurement_consist = measurement
    return measurement_consist, measurement_over, measurement_under


def prepare_light_measurements(
    measurement,
    measurement1,
    pipeline1,
    pipeline2,
    iterations=DEFAULT_LIGHT_ITERATIONS,
    strategy=DEFAULT_LIGHT_STRATEGY,
):
    modes1 = getattr(pipeline1, "exposure_modes", lambda: {"lowlight": False, "overexposure": False})()
    modes2 = getattr(pipeline2, "exposure_modes", lambda: {"lowlight": False, "overexposure": False})()
    light1 = modes1["lowlight"] or modes1["overexposure"]
    light2 = modes2["lowlight"] or modes2["overexposure"]
    if not light1 and not light2:
        return measurement, measurement, measurement, measurement1, measurement1, measurement1

    consist1, over1, under1 = measurement, measurement, measurement
    consist2, over2, under2 = measurement1, measurement1, measurement1
    if light1:
        consist1, over1, under1 = Get_Light_params(
            measurement,
            use_lowlight=modes1["lowlight"],
            use_overexposure=modes1["overexposure"],
            iterations=iterations,
            strategy=strategy,
        )
    if light2:
        consist2, over2, under2 = Get_Light_params(
            measurement1,
            use_lowlight=modes2["lowlight"],
            use_overexposure=modes2["overexposure"],
            iterations=iterations,
            strategy=strategy,
        )
    return consist1, over1, under1, consist2, over2, under2

def register_solver(name: str):
    def wrapper(cls):
        if __SOLVER__.get(name, None) is not None:
            raise ValueError(f"Solver {name} already registered.")
        __SOLVER__[name] = cls
        return cls
    return wrapper

def get_solver(name: str, **kwargs):
    if name not in __SOLVER__:
        raise ValueError(f"Solver {name} does not exist.")
    return __SOLVER__[name](**kwargs)

class StableDiffusion():
    def __init__(self,
                 solver_config: Dict,
                 model_key:str="./pretrained/stable-diffusion-v1-5/",
                 device: Optional[torch.device]=None,
                 **kwargs):
        self.device = device

        is_cpu = str(device).startswith("cpu")
        self.dtype = kwargs.get("pipe_dtype", torch.float32 if is_cpu else torch.float16)
        pipe = StableDiffusionPipeline.from_pretrained(model_key, torch_dtype=self.dtype).to(device)
        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet

        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
        total_timesteps = len(self.scheduler.timesteps)
        self.scheduler.set_timesteps(solver_config["num_sampling"], device=device)
        self.skip = total_timesteps // solver_config["num_sampling"]

        self.final_alpha_cumprod = self.scheduler.final_alpha_cumprod.to(device=device, dtype=self.dtype)
        self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(device=device, dtype=self.dtype)

    def alpha(self, t):
        index = int(t.item()) if torch.is_tensor(t) else int(t)
        if index >= 0:
            return self.scheduler.alphas_cumprod[index]
        return self.final_alpha_cumprod

    @torch.no_grad()
    def get_text_embed(self, null_prompt, prompt):
        """
        Get text embedding.
        args:
            null_prompt (str): null text
            prompt (str): guidance text
        """
        # null text embedding (negation)
        null_text_input = self.tokenizer(null_prompt,
                                         padding='max_length',
                                         max_length=self.tokenizer.model_max_length,
                                         return_tensors="pt",)
        null_text_embed = self.text_encoder(null_text_input.input_ids.to(self.device))[0]

        # text embedding (guidance)
        text_input = self.tokenizer(prompt,
                                    padding='max_length',
                                    max_length=self.tokenizer.model_max_length,
                                    return_tensors="pt",
                                    truncation=True)
        text_embed = self.text_encoder(text_input.input_ids.to(self.device))[0]

        return null_text_embed, text_embed

    def encode(self, x):
        """
        xt -> zt
        """
        x = x.to(device=self.device, dtype=self.dtype)
        return (self.vae.encode(x).latent_dist.sample() * 0.18215).to(dtype=self.dtype)

    def decode(self, zt):
        """
        zt -> xt
        """
        zt = zt.to(device=self.device, dtype=self.dtype)
        zt = 1/0.18215 * zt
        img = self.vae.decode(zt).sample.float()
        return img

    def predict_noise(self,
                      zt: torch.Tensor,
                      t: torch.Tensor,
                      uc: torch.Tensor,
                      c: torch.Tensor):
        """
        compuate epsilon_theta for null and condition
        args:
            zt (torch.Tensor): latent features
            t (torch.Tensor): timestep
            uc (torch.Tensor): null-text embedding
            c (torch.Tensor): text embedding
        """
        zt = zt.to(device=self.device, dtype=self.dtype)
        if uc is not None:
            uc = uc.to(device=self.device, dtype=self.dtype)
        if c is not None:
            c = c.to(device=self.device, dtype=self.dtype)
        if t.dim() == 0:
            t = t.unsqueeze(0)
        t = t.to(self.device)
        if uc is None:
            noise_c = self.unet(zt, t, encoder_hidden_states=c)['sample']
            noise_uc = noise_c
        elif c is None:
            noise_uc = self.unet(zt, t, encoder_hidden_states=uc)['sample']
            noise_c = noise_uc
        else:
            c_embed = torch.cat([uc, c], dim=0)
            z_in = torch.cat([zt] * 2) 
            t_in = torch.cat([t] * 2)
            noise_pred = self.unet(z_in, t_in, encoder_hidden_states=c_embed)['sample']
            noise_uc, noise_c = noise_pred.chunk(2)

        return noise_uc.to(dtype=self.dtype), noise_c.to(dtype=self.dtype)

    def initialize_latent(self, src_img: Optional[torch.Tensor] = None):
        if src_img is not None:
            src_img = src_img.clamp(0, 1).to(self.dtype).to(self.device)
            z_shape = self.encode(src_img * 2 - 1).shape
            z = torch.randn(z_shape, device=self.device, dtype=self.dtype)
        else:
            z = torch.randn((1, 4, 64, 64), device=self.device, dtype=self.dtype)
        return z.requires_grad_()

def conjugate_gradient_solver(A_op, rhs, x_init, tol=1e-6, max_iter=100):
    x = x_init.clone()
    r = rhs - A_op(x)
    p = r.clone()
    rsold = torch.sum(r * r)
    for i in range(max_iter):
        Ap = A_op(p)
        denom = torch.sum(p * Ap)
        if (not torch.isfinite(denom)) or denom.abs() < 1e-12:
            break
        alpha = rsold / denom
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = torch.sum(r * r)
        if torch.sqrt(rsnew) < tol:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    return x
def Get_background_system_params(x0t, measurement_target, A, At, B, Bt, weight_a, weight_b, damping=1e-4):
    def C(x):
        return weight_a * A(x) + weight_b * B(x)

    def Ct(y):
        return At(weight_a * y) + Bt(weight_b * y)

    b = Ct(measurement_target) + damping * x0t

    def A_op(x):
        return Ct(C(x)) + damping * x

    return conjugate_gradient_solver(A_op, b, ((x0t)), tol=1e-6, max_iter=10)

def Get_texture_system_params(
    x0y_background,
    diff_measurement,
    measurement_under,
    measurement_over,
    measurement,
    measurement1,
    measurement1_under,
    measurement1_over,
    A,
    At,
    B,
    Bt,
    weight_a,
    weight_b,
    damping=1e-4,
):
    S = x0y_background
    
    kernel_x, kernel_y = make_sobel_kernels(measurement.device,measurement.shape[1])
    lap_kernel = make_laplacian_kernel(measurement.device,measurement.shape[1])
    
    Gx_t1, Gy_t1 = image_gradients(measurement, kernel_x, kernel_y)
    Gx_t1_under, Gy_t1_under = image_gradients(measurement_under, kernel_x, kernel_y)
    Gx_t1_over, Gy_t1_over = image_gradients(measurement_over, kernel_x, kernel_y)    
    Gx_t2, Gy_t2 = image_gradients(measurement1, kernel_x, kernel_y)
    Gx_t2_under, Gy_t2_under = image_gradients(measurement1_under, kernel_x, kernel_y)
    Gx_t2_over, Gy_t2_over = image_gradients(measurement1_over, kernel_x, kernel_y)
    
    Gx_target = torch.where(torch.abs(Gx_t1) >= torch.abs(Gx_t2), Gx_t1, Gx_t2)
    Gx_target =(0*Gx_target+ 10*torch.where(torch.abs(Gx_target) >= torch.abs(Gx_t1_under), Gx_target, Gx_t1_under))/10
    Gx_target =(0*Gx_target+ 10*torch.where(torch.abs(Gx_target) >= torch.abs(Gx_t1_over), Gx_target, Gx_t1_over))/10
    Gx_target =(0*Gx_target+ 10*torch.where(torch.abs(Gx_target) >= torch.abs(Gx_t2_under), Gx_target, Gx_t2_under))/10
    Gx_target =(0*Gx_target+ 10*torch.where(torch.abs(Gx_target) >= torch.abs(Gx_t2_over), Gx_target, Gx_t2_over))/10
   
    Gy_target = torch.where(torch.abs(Gy_t1) >= torch.abs(Gy_t2), Gy_t1, Gy_t2)
    Gy_target = (0*Gy_target+ 10*torch.where(torch.abs(Gy_target) >= torch.abs(Gy_t1_under), Gy_target, Gy_t1_under))/10
    Gy_target = (0*Gy_target+ 10*torch.where(torch.abs(Gy_target) >= torch.abs(Gy_t1_over), Gy_target, Gy_t1_over))/10
    Gy_target = (0*Gy_target+ 10*torch.where(torch.abs(Gy_target) >= torch.abs(Gy_t2_under), Gy_target, Gy_t2_under))/10
    Gy_target = (0*Gy_target+ 10*torch.where(torch.abs(Gy_target) >= torch.abs(Gy_t2_over), Gy_target, Gy_t2_over))/10

    weight_x_a = (torch.abs(Gx_t1) >= torch.abs(Gx_t2)).to(measurement.dtype)
    weight_x_b = 1.0 - weight_x_a
    weight_y_a = (torch.abs(Gy_t1) >= torch.abs(Gy_t2)).to(measurement.dtype)
    weight_y_b = 1.0 - weight_y_a

    S_A = A(S)
    S_B = B(S)
    Gx_S_A, Gy_S_A = image_gradients(S_A, kernel_x, kernel_y)
    Gx_S_B, Gy_S_B = image_gradients(S_B, kernel_x, kernel_y)

    b_x = Gx_target - (weight_x_a * Gx_S_A + weight_x_b * Gx_S_B)
    b_y = Gy_target - (weight_y_a * Gy_S_A + weight_y_b * Gy_S_B)
    b_x_S = At(F.conv_transpose2d(weight_x_a * b_x, kernel_x, padding=1, groups=b_x.shape[1]))
    b_x_S = b_x_S + Bt(F.conv_transpose2d(weight_x_b * b_x, kernel_x, padding=1, groups=b_x.shape[1]))
    b_y_S = At(F.conv_transpose2d(weight_y_a * b_y, kernel_y, padding=1, groups=b_y.shape[1]))
    b_y_S = b_y_S + Bt(F.conv_transpose2d(weight_y_b * b_y, kernel_y, padding=1, groups=b_y.shape[1]))

    def C_ref(D):
        return weight_a * A(D) + weight_b * B(D)

    def Ct_ref(y):
        return At(weight_a * y) + Bt(weight_b * y)
 
    def A_op(D):
        D_A = A(D)
        D_B = B(D)
        conv_x = weight_x_a * F.conv2d(D_A, kernel_x, padding=1, groups=D.shape[1])
        conv_x = conv_x + weight_x_b * F.conv2d(D_B, kernel_x, padding=1, groups=D.shape[1])
        conv_y = weight_y_a * F.conv2d(D_A, kernel_y, padding=1, groups=D.shape[1])
        conv_y = conv_y + weight_y_b * F.conv2d(D_B, kernel_y, padding=1, groups=D.shape[1])
        At_conv_x = At(F.conv_transpose2d(weight_x_a * conv_x, kernel_x, padding=1, groups=D.shape[1]))
        At_conv_x = At_conv_x + Bt(F.conv_transpose2d(weight_x_b * conv_x, kernel_x, padding=1, groups=D.shape[1]))
        At_conv_y = At(F.conv_transpose2d(weight_y_a * conv_y, kernel_y, padding=1, groups=D.shape[1]))
        At_conv_y = At_conv_y + Bt(F.conv_transpose2d(weight_y_b * conv_y, kernel_y, padding=1, groups=D.shape[1]))
      
        D_ref = C_ref(D)
        lap = F.conv2d(D_ref, lap_kernel, padding=1, groups=D.shape[1])
        At_lap = F.conv_transpose2d(lap, lap_kernel, padding=1, groups=D.shape[1])
        At_ref = Ct_ref(D_ref)
        At_lap = Ct_ref(At_lap)
        return At_conv_x + At_conv_y + 0.3 * At_ref + 0.3 * At_lap + damping * D


    D_init = diff_measurement
    rhs = b_x_S + b_y_S + damping * D_init
    
    
    D_opt = conjugate_gradient_solver(A_op, rhs, D_init, tol=1e-6, max_iter=10)
   
    Y = x0y_background + D_opt
    return Y
   
   
@register_solver(name='PDFuse')
class PDFuse(StableDiffusion):
    @torch.no_grad()
    def data_consistency(
        self,
        z0t: torch.Tensor,
        measurement_under: torch.Tensor,
        measurement_over: torch.Tensor,
        measurement_consist: torch.Tensor,
        measurement: torch.Tensor,
        measurement1: torch.Tensor,
        measurement1_under: torch.Tensor,
        measurement1_over: torch.Tensor,
        at_prev: torch.Tensor,
        A: Callable,
        At: Callable,
        B: Callable,
        Bt: Callable,
        fusion_task: str,
        cg_damping: float = 1e-4,
    ):
        x0t = ((self.decode(z0t) + 1) / 2).detach()
        source_target, weight_a, weight_b = build_pixel_target_and_weights(
            measurement_consist, measurement1, fusion_task
        )
        measurement_target = (1 - at_prev) * source_target + at_prev * x0t

        x0y_background = Get_background_system_params(
            x0t, measurement_target, A, At, B, Bt, weight_a, weight_b, damping=cg_damping
        )
        diff_measurement = At(weight_a * measurement_target) + Bt(weight_b * measurement_target) - x0y_background
        x0y = Get_texture_system_params(
            x0y_background,
            diff_measurement,
            measurement_under,
            measurement_over,
            measurement,
            measurement1,
            measurement1_under,
            measurement1_over,
            A,
            At,
            B,
            Bt,
            weight_a,
            weight_b,
            damping=cg_damping,
        )

        x0y_background = gaussian_blur(x0y)
        z0y = (1 - at_prev) * self.encode(x0y_background * 2 - 1) + at_prev * z0t
        return z0y, (at_prev)*(x0y-x0y_background)

    def sample(self,
               measurement: torch.Tensor,
               measurement1: torch.Tensor,
               prompt: list[str]=["", [""]],
               **kwargs):

        no_dps = kwargs.get("no_dps", lambda step: True)
        fusion_task = kwargs.get("fusion_task", "IVF")
        return_observations = kwargs.get("return_observations", False)
        inputs_are_observed = kwargs.get("inputs_are_observed", False)
        pipeline1 = kwargs.get("operator1")
        pipeline2 = kwargs.get("operator2")
        if pipeline1 is None:
            pipeline1 = build_degradation_pipeline(kwargs.get("degradation1", "none"))
        if pipeline2 is None:
            pipeline2 = build_degradation_pipeline(kwargs.get("degradation2", "none"))
        light_iterations = kwargs.get("light_iterations", DEFAULT_LIGHT_ITERATIONS)
        light_strategy = kwargs.get("light_strategy", DEFAULT_LIGHT_STRATEGY)
        cg_damping = kwargs.get("cg_damping", 1e-4)

        operator_pipeline1 = strip_exposure_operators(pipeline1)
        operator_pipeline2 = strip_exposure_operators(pipeline2)
        if inputs_are_observed:
            measurement_observed = measurement
            measurement1_observed = measurement1
        else:
            measurement_observed = operator_pipeline1(measurement)
            measurement1_observed = operator_pipeline2(measurement1)

        (
            measurement_consist,
            measurement_over,
            measurement_under,
            measurement1_consist,
            measurement1_over,
            measurement1_under,
        ) = prepare_light_measurements(
            measurement_observed,
            measurement1_observed,
            pipeline1,
            pipeline2,
            iterations=light_iterations,
            strategy=light_strategy,
        )
        A, At = operator_pipeline1, operator_pipeline1.transpose
        B, Bt = operator_pipeline2, operator_pipeline2.transpose

        # Text embedding
        tgt_texts = prompt[1]
        c_tgt_list = [self.get_text_embed(null_prompt=prompt[0], prompt=tgt)[1] for tgt in tgt_texts]
        tgt_c = torch.cat(c_tgt_list, dim=0)

        init_target, init_weight_a, init_weight_b = build_pixel_target_and_weights(
            measurement_consist, measurement1_consist, fusion_task
        )
        # Initialize zT
        zt = self.initialize_latent(
            src_img=At(init_weight_a * init_target) + Bt(init_weight_b * init_target),
        )


        measurement_target = build_pixel_target(measurement_consist, measurement1_consist, fusion_task)
        x0_diff = measurement_target - gaussian_blur(measurement_target)

        # Sampling
        pbar = tqdm(self.scheduler.timesteps, desc="PDFuse")
        start_time = time.time()
        for step, t in enumerate(pbar):
            at = self.alpha(t)
            at_prev = self.alpha(t - self.skip)
            if no_dps(step):
                with torch.no_grad():
                    if step ==0:
                        at = self.alpha(t)
                        at_prev = self.alpha(t - self.skip)
                        at_form = self.scheduler.alphas_cumprod[-1]
                        t_form=torch.tensor(999).to(zt.device)
                        _, noise_pred = self.predict_noise(zt, t_form, None, tgt_c)
                        z0t = (zt - (1-at_form).sqrt() * noise_pred) / at_form.sqrt()
                        zt = at.sqrt() * z0t + (1-at).sqrt() * noise_pred
                    else:
                        at = self.alpha(t)
                        at_prev = (
                            self.alpha(t - self.skip)
                            if step < len(self.scheduler.timesteps) - 1
                            else torch.tensor(1.0, device=zt.device, dtype=at.dtype)
                        )
                    _, noise_pred = self.predict_noise(zt, t, None, tgt_c)
                    z0t = (zt - (1-at).sqrt() * noise_pred) / at.sqrt()
                    z0y, x0_diff = self.data_consistency(
                        z0t,
                        measurement_under,
                        measurement_over,
                        measurement_consist,
                        measurement_consist,
                        measurement1_consist,
                        measurement1_under,
                        measurement1_over,
                        at,
                        A,
                        At,
                        B,
                        Bt,
                        fusion_task,
                        cg_damping,
                    )
                    zt = at_prev.sqrt() * z0y + (1-at_prev).sqrt() * noise_pred
            else:
                with torch.enable_grad():
                    if step ==0:
                        at = self.alpha(t)
                        at_prev = self.alpha(t - self.skip)
                        at_form = self.scheduler.alphas_cumprod[-1]
                        t_form=torch.tensor(999).to(zt.device)
                        _, noise_pred = self.predict_noise(zt, t_form, None, tgt_c)
                        z0t = (zt - (1-at_form).sqrt() * noise_pred) / at_form.sqrt()
                        zt = at.sqrt() * z0t + (1-at).sqrt() * noise_pred
                    else:
                        at = self.alpha(t)
                        at_prev = (
                            self.alpha(t - self.skip)
                            if step < len(self.scheduler.timesteps) - 1
                            else torch.tensor(1.0, device=zt.device, dtype=at.dtype)
                        )
                    zt=zt.requires_grad_()
                    _, noise_pred = self.predict_noise(zt, t, None, tgt_c)
                    z0t = (zt - (1-at).sqrt() * noise_pred) / at.sqrt()
                    zt_prime = at_prev.sqrt() * z0t + (1-at_prev).sqrt() * noise_pred  
                    x0t = self.decode(z0t)
                    kernel_x, kernel_y = make_sobel_kernels(measurement_consist.device,measurement_consist.shape[1])
                    Gx_t1, Gy_t1 = image_gradients(measurement_consist, kernel_x, kernel_y)
                    Gx_t1_under, Gy_t1_under = image_gradients(measurement_under, kernel_x, kernel_y)
                    Gx_t1_over, Gy_t1_over = image_gradients(measurement_over, kernel_x, kernel_y)    
                    Gx_t2, Gy_t2 = image_gradients(measurement1_consist, kernel_x, kernel_y)
                    Gx_t2_under, Gy_t2_under = image_gradients(measurement1_under, kernel_x, kernel_y)
                    Gx_t2_over, Gy_t2_over = image_gradients(measurement1_over, kernel_x, kernel_y)
    
                    Gx_target = torch.where(torch.abs(Gx_t1) >= torch.abs(Gx_t2), Gx_t1, Gx_t2)
                    Gx_target =(0*Gx_target+ 10*torch.where(torch.abs(Gx_target) >= torch.abs(Gx_t1_under), Gx_target, Gx_t1_under))/10
                    Gx_target =(0*Gx_target+ 10*torch.where(torch.abs(Gx_target) >= torch.abs(Gx_t1_over), Gx_target, Gx_t1_over))/10
                    Gx_target =(0*Gx_target+ 10*torch.where(torch.abs(Gx_target) >= torch.abs(Gx_t2_under), Gx_target, Gx_t2_under))/10
                    Gx_target =(0*Gx_target+ 10*torch.where(torch.abs(Gx_target) >= torch.abs(Gx_t2_over), Gx_target, Gx_t2_over))/10
   
                    Gy_target = torch.where(torch.abs(Gy_t1) >= torch.abs(Gy_t2), Gy_t1, Gy_t2)
                    Gy_target = (0*Gy_target+ 10*torch.where(torch.abs(Gy_target) >= torch.abs(Gy_t1_under), Gy_target, Gy_t1_under))/10
                    Gy_target = (0*Gy_target+ 10*torch.where(torch.abs(Gy_target) >= torch.abs(Gy_t1_over), Gy_target, Gy_t1_over))/10
                    Gy_target = (0*Gy_target+ 10*torch.where(torch.abs(Gy_target) >= torch.abs(Gy_t2_under), Gy_target, Gy_t2_under))/10
                    Gy_target = (0*Gy_target+ 10*torch.where(torch.abs(Gy_target) >= torch.abs(Gy_t2_over), Gy_target, Gy_t2_over))/10
                    x0t_img = ((x0t + 1) / 2).float()
                    measurement_target, weight_a, weight_b = build_pixel_target_and_weights(
                        measurement_consist, measurement1_consist, fusion_task
                    )
                    x0t_A = weight_a * A(x0t_img) + weight_b * B(x0t_img)
                  
                    Gx_x0t_A,Gy_x0t_A=image_gradients(x0t_A, kernel_x, kernel_y)
                    
                   
                    residue = torch.linalg.norm((measurement_target - x0t_A).reshape(-1))
                    residue = residue + torch.linalg.norm((Gx_target - Gx_x0t_A).reshape(-1))
                    residue = residue + torch.linalg.norm((Gy_target - Gy_x0t_A).reshape(-1))
                    projected_target = At(weight_a * measurement_target) + Bt(weight_b * measurement_target)
                    projected_current = At(weight_a * x0t_A) + Bt(weight_b * x0t_A)
                    latent_anchor = gaussian_blur(projected_target - x0t_img + projected_current).clamp(0, 1)
                    residue_latent=torch.linalg.norm(  ( self.encode(latent_anchor * 2 - 1) -z0t).reshape(-1))
                    grad = torch.autograd.grad(1*residue+1*residue_latent, zt,retain_graph=False)[0]
                    dps_lamb = at_prev.sqrt()
                    zt = zt_prime - dps_lamb * (grad)

            
                

        # for the last step, do not add noise
        with torch.no_grad():
            img = self.decode(zt)
        img = ((img / 2 + 0.5)+x0_diff).clamp(0, 1)
        if (
            measurement.shape[1] == 3
            and measurement1.shape[1] == 3
            and torch.allclose(measurement[:, 0:1, :, :], measurement[:, 1:2, :, :])
            and torch.allclose(measurement[:, 1:2, :, :], measurement[:, 2:3, :, :])
            and torch.allclose(measurement1[:, 0:1, :, :], measurement1[:, 1:2, :, :])
            and torch.allclose(measurement1[:, 1:2, :, :], measurement1[:, 2:3, :, :])
        ):
            img = img.mean(dim=1, keepdim=True).repeat(1, img.shape[1], 1, 1)
        end_time = time.time()
        print(f"PDFuse sampling time: {end_time - start_time:.2f}s")
        img = img.detach().cpu()
        if return_observations:
            return {
                "recon": img.float(),
                "observed1": measurement_observed.detach().float().cpu(),
                "observed2": measurement1_observed.detach().float().cpu(),
                "measurement1_consist": measurement_consist.detach().float().cpu(),
                "measurement2_consist": measurement1_consist.detach().float().cpu(),
            }
        return img.float()
