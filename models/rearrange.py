import paddle


def rearrange(x, mode: int, h_new=None):
    if mode == 0:  # 'b c h w -> b (h w) c'
        b, c, h, w = x.shape
        x = paddle.transpose(x, perm=[0, 2, 3, 1])
        x = paddle.reshape(x, shape=[b, h * w, c])
    if mode == 1:  # 'b (h w) c -> b c h w'
        b, hw, c = x.shape
        x = paddle.reshape(x, shape=[b, h_new, hw // h_new, c])
        x = paddle.transpose(x, perm=[0, 3, 1, 2])
    if mode == 2:  # 'b t (h d) -> b h t d'
        b, t, hd = x.shape
        x = paddle.reshape(x, shape=[b, t, h_new, hd // h_new])
        x = paddle.transpose(x, perm=[0, 2, 1, 3])
    if mode == 3:  # 'b h t d -> b t (h d)'
        b, h, t, d = x.shape
        x = paddle.transpose(x, perm=[0, 2, 1, 3])
        x = paddle.reshape(x, shape=[b, t, h * d])
    return x


class Rearrange(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        b, c, h, w = x.shape
        x = paddle.transpose(x, perm=[0, 2, 3, 1])
        x = paddle.reshape(x, shape=[b, h * w, c])
        return x
