_base_ = ['./oneformer3d_qs_radius16_qp300_2many.py']


model = dict(
    qps_fps_mode='ogrd',
    # OGRD learnable alpha initialization value.
    qps_ogrd_alpha=0.1,
)

vis_backends = [
                dict(
                    type='WandbVisBackend',
                    init_kwargs=dict(
                        entity='wuhaili2002-cas',
                        project='ForestFormer3D',
                        name='fps_ogrd',
                        mode='offline'
            ))]