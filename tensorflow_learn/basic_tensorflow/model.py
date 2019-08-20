"""
实现模型持久化
保存的3个文件全为二进制文件:
model.ckpt.data-***文件保存了Tensorflow程序中每一个变量的取值
model.ckpt.index文件保存了每一个变量的名称,是一个string-string的table,其中table的key值为tensor名,value值为捆绑入口原型
model.ckpt.meta文件保存了计算图的结构,或者说是神经网络的结构
"""
import os
import tensorflow as tf


class Model(object):
    def __init__(self):
        # 定义项目目录,模型保存目录, 模型保存路径
        self.project_dir = '/home/xiaonan/python_project/tensorflow_learn/'
        self.model_dir = 'basic_tensorflow/model/'
        self.model_path = os.path.join(self.project_dir, self.model_dir)
        # 若模型保存目录不存在,则创建
        if os.path.exists(self.model_path) is False:
            os.mkdir(self.model_path)

    def model_save(self):
        # 声明两个向量并相加
        a = tf.Variable(tf.constant([1.0, 2.0], tf.float32, [2]), name='a')
        b = tf.Variable(tf.constant([3.0, 4.0], dtype=tf.float32, shape=[2]), name='b')
        c = a+b

        # 定义Saver类对象用于保存模型
        saver = tf.compat.v1.train.Saver()
        saver.export_meta_graph(self.model_path + '/model_ckpt_meta_json', as_text=True)

        # 开启会话,进行图计算
        with tf.compat.v1.Session() as sess:
            # 初始化所有变量
            tf.compat.v1.global_variables_initializer().run()
            # 从session中保存计算图中的参数模型,使用save()函数进行保存
            saver.save(sess, os.path.join(self.model_path, 'model.ckpt'))

    def model_restore(self):
        # 定义恢复的计算图
        a2 = tf.Variable(tf.constant([15.0, 2.0], tf.float32, [2]), name='a2')
        b2 = tf.Variable(tf.constant([3.0, 4.0], tf.float32, [2]), name='b2')
        c2 = a2 + b2

        # 定义Saver类模型,恢复a
        saver = tf.compat.v1.train.Saver({'a': a2, 'b':b2})

        # # 文件路径
        # meta_file = os.path.join(self.model_path, 'model.ckpt.meta')
        # # 省略了定义计算图上运算的过程,使用.meta文件直接加载持久化的计算图
        # meta_graph = tf.compat.v1.train.import_meta_graph(meta_file)

        # 开启session
        with tf.compat.v1.Session() as sess:
            # # 使用restore()函数恢复模型
            # meta_graph.restore(sess, os.path.join(self.model_path, 'model.ckpt'))
            # # 获得默认计算图上指定节点处的张量的值
            # print(sess.run(tf.compat.v1.get_default_graph().get_tensor_by_name('add:0')))

            # get_checkpoint_state()函数会通过checkpoint文件自动找到目录中最新模型的文件名
            ckpt = tf.compat.v1.train.get_checkpoint_state(self.model_path)
            # 恢复模型
            saver.restore(sess, ckpt.model_checkpoint_path)
            print(a2, b2, c2)
            print(sess.run([a2, b2, c2]))

    def read_model(self):
        # NewCheckpointReader()函数读取checkpoint文件中最新保存的.index和.data文件
        reader = tf.compat.v1.train.NewCheckpointReader(filepattern=os.path.join(self.model_path, 'model.ckpt'))
        # 获取文件中的所有变量列表,以字典的形式,即{变量名:形状}
        all_variable = reader.get_variable_to_shape_map()
        print(all_variable)
        # 根据变量名获得张量值,形状
        for var_name, shape in all_variable.items():
            print(var_name, reader.get_tensor(var_name), shape)

    def save_to_pb_file(self):
        """
        将模型保存成pb格式
        :return:
        """
        a = tf.Variable(tf.constant([1.0, 2.0], tf.float32, [2]), name='a')
        b = tf.Variable(tf.constant([3.0, 4.0], tf.float32, [2]), name='b')
        c = a + b

        with tf.compat.v1.Session() as sess:
            tf.compat.v1.global_variables_initializer().run()

            # 使用get_default_graph()函数获得默认的计算图,再使用as_graph_def()函数获得计算图节点信息的GraphDef部分
            graph_def = tf.compat.v1.get_default_graph().as_graph_def()

            # convert_variables_to_constants()函数用相同值的常量替换计算图的所有变量
            out_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(sess, graph_def, ['add'])

            # 将导出的模型存入.pb文件
            with tf.io.gfile.GFile(os.path.join(self.model_path, 'model.pb'), 'wb') as file:
                # SerializeToString()函数用于将获取到的数据取出存到一个string对象中,然后再以二进制流的方式将其写入到
                file.write(out_graph_def.SerializeToString())

            # 读取.pb文件中的数据
            with tf.gfile.FastGFile(os.path.join(self.model_path, 'model.pb'), 'rb') as file:
                graph_def2 = tf.compat.v1.GraphDef()
                # 使用FastGFile类的read()函数读取保存的模型文件,并以字符串形式返回文件的内容
                # 再使用ParseFromString()函数解析字符串的内容
                graph_def2.ParseFromString(file.read())
                # 使用import_graph_def()函数将graph_def中保存的计算图加载到当前图中
                result = tf.import_graph_def(graph_def2, return_elements=["add:0"])
                print(sess.run(result))


if __name__ == '__main__':
    model = Model()
    model.model_save()
    model.model_restore()
    model.read_model()
    model.save_to_pb_file()