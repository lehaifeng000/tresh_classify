class Convolution:
    def __init__(self, W, fb, stride=1, pad=0):
        """
        W-- 滤波器权重，shape为(FN,NC,FH,FW),FN 为滤波器的个数
        fb -- 滤波器的偏置，shape 为(1,FN)
        stride -- 步长
        pad -- 填充个数
        """
        self.W = W
        self.fb = fb
        self.stride = stride
        self.pad = pad

        self.col_X = None
        self.X = None
        self.col_W = None

        self.dW = None
        self.db = None
        self.out_shape = None

    #    self.out = None

    def forward(self, input_X):
        """
        input_X-- shape为(m,nc,height,width)
        """
        self.X = input_X
        FN, NC, FH, FW = self.W.shape

        m, input_nc, input_h, input_w = self.X.shape

        # 先计算输出的height和widt
        out_h = int((input_h + 2 * self.pad - FH) / self.stride + 1)
        out_w = int((input_w + 2 * self.pad - FW) / self.stride + 1)

        # 将输入数据展开成二维数组，shape为（m*out_h*out_w,FH*FW*C)
        self.col_X = col_X = im2col2(self.X, FH, FW, self.stride, self.pad)

        # 将滤波器一个个按列展开(FH*FW*C,FN)
        self.col_W = col_W = self.W.reshape(FN, -1).T
        out = np.dot(col_X, col_W) + self.fb
        out = out.T
        out = out.reshape(m, FN, out_h, out_w)
        self.out_shape = out.shape
        return out
    #
    # def backward(self, dz, learning_rate):
    #     # print("==== Conv backbward ==== ")
    #     assert (dz.shape == self.out_shape)
    #
    #     FN, NC, FH, FW = self.W.shape
    #     o_FN, o_NC, o_FH, o_FW = self.out_shape
    #
    #     col_dz = dz.reshape(o_NC, -1)
    #     col_dz = col_dz.T
    #
    #     self.dW = np.dot(self.col_X.T, col_dz)  # shape is (FH*FW*C,FN)
    #     self.db = np.sum(col_dz, axis=0, keepdims=True)
    #
    #     self.dW = self.dW.T.reshape(self.W.shape)
    #     self.db = self.db.reshape(self.fb.shape)
    #
    #     d_col_x = np.dot(col_dz, self.col_W.T)  # shape is (m*out_h*out_w,FH,FW*C)
    #     dx = col2im2(d_col_x, self.X.shape, FH, FW, stride=1)
    #
    #     assert (dx.shape == self.X.shape)
    #
    #     # 更新W和b
    #     self.W = self.W - learning_rate * self.dW
    #     self.fb = self.fb - learning_rate * self.db
    #
    #     return dx
