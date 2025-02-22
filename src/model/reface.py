from functools import partial

import dlib
import numpy as np
import torch
from torch import nn
from torchvision.transforms import Resize
from tqdm import tqdm

from .ddpm import DDPM
from .face_model.arcface import Arcface
from .ldm.utils import (
    denormalize,
    extract_into_tensor,
    make_beta_schedule,
    make_ddim_sampling_parameters,
    make_ddim_timesteps,
    noise_like,
    require_grad,
    un_norm_clip,
)


class REFace(DDPM):
    def __init__(
        self,
        model,
        vae,
        device,
        landmark_predictor_path,
        arcface_path,
        ddim_steps,
    ):
        super().__init__(model)
        self.model = model
        self.vae = vae
        self.detector = dlib.get_frontal_face_detector()
        self.device = device
        self.predictor = dlib.shape_predictor(landmark_predictor_path)
        self.landmark_proj_out = nn.Linear(136, 256)
        self.last_proj = nn.Linear(768, 768)
        self.instantiate_id_model(arcface_path)
        self.ddim_steps = ddim_steps
        self.make_schedule(ddim_num_steps=self.ddim_steps)

    def instantiate_id_model(self, path):
        model = Arcface(path, multiscale=False)
        self.face_ID_model = model.eval()
        require_grad(self.face_ID_model, False)

    @torch.no_grad()
    def encode(self, x):
        if x is not None:
            return self.vae.encode(x).latent_dist.sample().mul_(0.18215)
        return None

    def decode(self, x):
        if x is not None:
            return self.vae.decode(x / 0.18215).sample
        return None

    @torch.no_grad()
    def get_landmarks(self, x):
        if x is not None:
            x = 255.0 * denormalize(x).permute(0, 2, 3, 1).cpu().numpy()
            x = x.astype(np.uint8)
            landmarks = []
            for i in range(len(x)):
                faces = self.detector(x[i], 1)
                if len(faces) == 0:
                    landmarks.append(torch.zeros(1, 136))
                    continue
                shape = self.predictor(x[i], faces[0])
                t = list(shape.parts())
                a = []
                for tt in t:
                    a.append([tt.x, tt.y])
                lm = np.array(a)
                lm = lm.reshape(1, 136)
                landmarks.append(lm)
            landmarks = np.concatenate(landmarks, axis=0)
            landmarks = torch.tensor(landmarks).float().to(self.device)
            return self.landmark_proj_out(landmarks)

    @torch.no_grad()
    def get_condition(self, target_img, source_img):
        landmarks = self.get_landmarks(target_img)
        c_landmark = landmarks.unsqueeze(1) if len(landmarks.shape) != 3 else landmarks
        c_id = self.face_ID_model.extract_feats(source_img)[0].unsqueeze(1)
        c_all = torch.cat([c_landmark, c_id], dim=-1)
        return self.last_proj(c_all)

    @torch.no_grad()
    def prepare_batch(
        self, target_img, inpaint_img, mask, corrupt_img, source_img, **batch
    ):
        target_latent = self.encode(target_img)
        inpaint_latent = self.encode(inpaint_img)
        if corrupt_img is not None:
            corrupt_latent = self.encode(corrupt_img)
        else:
            corrupt_latent = None
        mask = Resize([target_latent.shape[-1], target_latent.shape[-1]])(mask)
        condition = self.get_condition(target_img, source_img)
        return {
            "target_img_latent": target_latent,
            "inpaint_img_latent": inpaint_latent,
            "corrupt_img_latent": corrupt_latent,
            "mask_resize": mask,
            "condition": condition,
        }

    def apply_model(self, x, t, cond):
        return self.model(x, t, context=cond.unsqueeze(1))

    def forward(self, **batch):
        output = self.prepare_batch(**batch)
        batch.update(output)
        t = torch.randint(
            0,
            self.num_timesteps,
            (batch["target_img_latent"].shape[0],),
            device=self.device,
        ).long()
        noise = torch.randn_like(batch["target_img_latent"])
        noise_img = self.q_sample(x_start=batch["target_img_latent"], t=t, noise=noise)
        if batch["corrupt_img_latent"] is not None:
            noise_img = torch.cat(
                [
                    noise_img,
                    batch["inpaint_img_latent"],
                    batch["mask_resize"],
                    batch["corrupt_img_latent"],
                ],
                dim=1,
            )
        else:
            noise_img = torch.cat(
                [noise_img, batch["inpaint_img_latent"], batch["mask_resize"]], dim=1
            )
        predict = self.apply_model(noise_img, t, batch["condition"])
        return {"predict": predict}

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            attr = attr.to(self.device)
        setattr(self, name, attr)

    def make_schedule(
        self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0.0, verbose=True
    ):
        self.ddim_timesteps = make_ddim_timesteps(
            ddim_discr_method=ddim_discretize,
            num_ddim_timesteps=ddim_num_steps,
            num_ddpm_timesteps=self.timesteps,
            verbose=verbose,
        )
        alphas_cumprod = self.model.alphas_cumprod
        assert (
            alphas_cumprod.shape[0] == self.timesteps
        ), "alphas have to be defined for each timestep"

        def to_torch(x):
            return x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer("ddim_betas", to_torch(self.model.betas))
        self.register_buffer("ddim_alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer(
            "ddim_alphas_cumprod_prev", to_torch(self.model.alphas_cumprod_prev)
        )

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            "ddim_sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod.cpu()))
        )
        self.register_buffer(
            "ddim_sqrt_one_minus_alphas_cumprod",
            to_torch(np.sqrt(1.0 - alphas_cumprod.cpu())),
        )
        self.register_buffer(
            "ddim_log_one_minus_alphas_cumprod",
            to_torch(np.log(1.0 - alphas_cumprod.cpu())),
        )
        self.register_buffer(
            "ddim_sqrt_recip_alphas_cumprod",
            to_torch(np.sqrt(1.0 / alphas_cumprod.cpu())),
        )
        self.register_buffer(
            "ddim_sqrt_recipm1_alphas_cumprod",
            to_torch(np.sqrt(1.0 / alphas_cumprod.cpu() - 1)),
        )

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(
            alphacums=alphas_cumprod.cpu(),
            ddim_timesteps=self.ddim_timesteps,
            eta=ddim_eta,
            verbose=verbose,
        )
        self.register_buffer("ddim_sigmas", ddim_sigmas)
        self.register_buffer("ddim_alphas", ddim_alphas)
        self.register_buffer("ddim_alphas_prev", ddim_alphas_prev)
        self.register_buffer("ddim_sqrt_one_minus_alphas", np.sqrt(1.0 - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.ddim_alphas_cumprod_prev)
            / (1 - self.ddim_alphas_cumprod)
            * (1 - self.ddim_alphas_cumprod / self.ddim_alphas_cumprod_prev)
        )
        self.register_buffer(
            "ddim_sigmas_for_original_num_steps", sigmas_for_original_sampling_steps
        )

    @torch.no_grad()
    def sample(
        self,
        mask=None,
        x0=None,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        **batch,
    ):
        output = self.prepare_batch(**batch)
        batch.update(output)
        samples = self.ddim_sampling(
            mask=mask,
            x0=x0,
            ddim_use_original_steps=False,
            unconditional_guidance_scale=unconditional_guidance_scale,
            unconditional_conditioning=unconditional_conditioning,
            **batch,
        )
        return self.decode(samples)

    @torch.no_grad()
    def ddim_sampling(
        self,
        ddim_use_original_steps=False,
        x0=None,
        mask=None,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        **batch,
    ):
        device = self.device

        img = torch.randn_like(batch["inpaint_img_latent"])

        timesteps = self.timesteps if ddim_use_original_steps else self.ddim_timesteps

        time_range = (
            reversed(range(0, timesteps))
            if ddim_use_original_steps
            else np.flip(timesteps)
        )
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc="DDIM Sampler", total=total_steps)
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((img.shape[0],), step, device=device, dtype=torch.long)
            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(
                    x0, ts
                )  # TODO: deterministic forward pass?
                img = img_orig * mask + (1.0 - mask) * img
                print("here")
            img = self.p_sample_ddim(
                img,
                ts,
                index=index,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=unconditional_conditioning,
                **batch,
            )

        return img

    @torch.no_grad()
    def p_sample_ddim(
        self,
        x,
        t,
        condition,
        index,
        repeat_noise=False,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        **batch,
    ):
        b = x.shape[0]
        device = x.device
        if batch["corrupt_img_latent"] is not None:
            x = torch.cat(
                [
                    x,
                    batch["inpaint_img_latent"],
                    batch["mask_resize"],
                    batch["corrupt_img_latent"],
                ],
                dim=1,
            )
        else:
            x = torch.cat([x, batch["inpaint_img_latent"], batch["mask_resize"]], dim=1)
        if unconditional_conditioning is None or unconditional_guidance_scale == 1.0:
            e_t = self.apply_model(x, t, condition)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            c_in = torch.cat([unconditional_conditioning, condition])
            e_t_uncond, e_t = self.apply_model(x_in, t_in, c_in).chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

        alphas = self.ddim_alphas
        alphas_prev = self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.ddim_sqrt_one_minus_alphas
        sigmas = self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full(
            (b, 1, 1, 1), sqrt_one_minus_alphas[index], device=device
        )

        # current prediction for x_0
        if x.shape[1] != 4:
            pred_x0 = (x[:, :4, :, :] - sqrt_one_minus_at * e_t) / a_t.sqrt()
        else:
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        dir_xt = (1.0 - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(dir_xt.shape, device, repeat_noise)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0
