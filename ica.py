import numpy as np

eps = 1e-4


class ICA:
    def __init__(self, x):
        self.x = np.matrix(x)

    def ica(self):  # 独立成分分析
        self.fit()
        z = self.whiten()
        y = self.analize(z)
        return y

    def fit(self):  # 平均を0にする（中心化）
        self.x -= self.x.mean(axis=1)

    def whiten(self):  # 分散共分散行列を対角化（白色化）

        # 共分散行列をnp.covで求める(行が1データの組，分散は標本共分散)
        sigma = np.cov(self.x, rowvar=True, bias=True)

        # sigmaの固有値，固有ベクトルを求める（対称行列なのでeighを使う）
        D, E = np.linalg.eigh(sigma)

        # Eをmatrixとして扱う
        E = np.asmatrix(E)

        Dh = np.diag(np.array(D) ** (-1/2))
        V = E * Dh * E.T
        z = V * self.x
        return z

    def normalize(self, x):  # 正規化
        if x.sum() < 0:
            x *= -1
        return x / np.linalg.norm(x)

    def analize(self, z):  # 独立成分分析をおこなう
        c, r = self.x.shape
        W = np.empty((0, c))
        for _ in range(c):  # 観測数分だけ実行
            vec_w = np.random.rand(c, 1)
            vec_w = self.normalize(vec_w)
            while True:
                vec_w_prev = vec_w
                vec_w = np.asmatrix((np.asarray(z) * np.asarray(vec_w.T * z) ** 3).mean(axis=1)).T - 3 * vec_w

                # 正規化＋直交化
                vec_w = self.normalize(np.linalg.qr(np.asmatrix(np.concatenate((W, vec_w.T), axis=0)).T)[0].T[-1].T)

                # 収束判定
                if np.linalg.norm(vec_w - vec_w_prev) < eps:
                    W = np.concatenate((W, vec_w.T), axis=0)
                    break
        y = W * z
        return y

