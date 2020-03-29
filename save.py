import shutil



class saveModule(object):
    def __init__(self, category='Mnist', z_size=100, lr=0.001):

        self.category = category
        self.z_size = z_size
        self.lr = lr


        self.src_dir = './output/'
        self.des_dir = './output/history/2000{}/z{}_lr{}'.format(
            self.category,
            str(self.z_size),
            str(self.lr))


    def process(self):
        lst = [
            'checkpoint',
            'gens',
            'logs',
        ]
        for dir in lst:
            src = self.src_dir + dir
            des = self.des_dir + '/' + dir
            shutil.copytree(src, des)
            shutil.rmtree(src)

