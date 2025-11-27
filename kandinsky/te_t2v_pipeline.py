import torch

from .t2v_pipeline import Kandinsky5T2VPipeline
from .models.te_dit import get_dit as get_te_dit


class Kandinsky5T2VTEPipeline(Kandinsky5T2VPipeline):
    """
    TE-enhanced Text-to-Video pipeline.

    Thin wrapper around the original `Kandinsky5T2VPipeline` that swaps the
    DiT backbone for a Transformer-Engine FP8 variant.

    Usage (Pro 10s SFT, HD):

        from omegaconf import OmegaConf
        from kandinsky.te_t2v_pipeline import Kandinsky5T2VTEPipeline
        from kandinsky.models.text_embedders import get_text_embedder
        from kandinsky.models.vae import get_vae

        conf = OmegaConf.load("configs/te_fp8_k5_pro_t2v_10s_sft_hd.yaml")

        text_embedder = get_text_embedder(conf)
        vae = get_vae(conf)

        device_map = {"dit": "cuda", "vae": "cuda", "text_embedder": "cuda"}

        pipe = Kandinsky5T2VTEPipeline.from_config(
            device_map=device_map,
            text_embedder=text_embedder,
            vae=vae,
            conf=conf,
            backend="te-fp8",
        )

        video = pipe(
            text="A cinematic description...",
            time_length=10,
            width=1024,
            height=576,
            seed=42,
        )
    """

    backend: str = "te-fp8"

    def __init__(self, *args, backend: str = "te-fp8", **kwargs):
        # We don't alter the base pipeline internals; we only record which
        # TE backend was used to construct the DiT.
        super().__init__(*args, **kwargs)
        self.backend = backend

    @classmethod
    def from_config(
        cls,
        device_map,
        text_embedder,
        vae,
        conf,
        backend: str = "te-fp8",
        local_dit_rank: int = 0,
        world_size: int = 1,
        offload: bool = False,
        device_mesh=None,
    ):
        """
        Convenience constructor that builds a TE DiT directly from the config.

        Parameters
        ----------
        device_map:
            Same as in the original pipeline
            (e.g. {"dit": "cuda", "vae": "cuda", "text_embedder": "cuda"}).
        text_embedder:
            Pre-built text embedder model.
        vae:
            Pre-built VAE model.
        conf:
            Full OmegaConf configuration (the same one you pass to the
            original pipeline). We expect `conf.model.dit_params` to exist.
        backend:
            Which TE backend to use. Currently only "te-fp8" is implemented.
        """
        dit_conf = conf.model.dit_params
        te_dit = get_te_dit(dit_conf, backend=backend)

        return cls(
            device_map=device_map,
            dit=te_dit,
            text_embedder=text_embedder,
            vae=vae,
            local_dit_rank=local_dit_rank,
            world_size=world_size,
            conf=conf,
            offload=offload,
            device_mesh=device_mesh,
            backend=backend,
        )
