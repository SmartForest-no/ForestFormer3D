_base_ = ['./oneformer3d_qs_radius16_qp300_2many.py']

model = dict(
    qps_fps_mode='tcfps',
    qps_tcfps_embed_ratio=0.6,
)
vis_backends = [
                dict(
                    type='WandbVisBackend',
                    init_kwargs=dict(
                        entity='wuhaili2002-cas',
                        project='ForestFormer3D',
                        name='fps_tcfps',
                        mode='offline'
            ))]
