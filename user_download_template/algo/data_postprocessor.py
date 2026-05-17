import utils  # (公共文件)


class data_postprocessor(object):
    def __init__(self, label_template_path, params_dict):
        """
            此类实现 模型预测输出 --> 指定的JSON数据格式(同数据管理平台单帧标签格式)
            构造函数内容不可删改,但可以自定义增加
            label_template_path: 标签模板label.json所对应的路径,用户可自行解析获取内容
            params_dict: algo_config.yaml文件中 TRAIN_MODEL模块对应的参数字典
            """
        self.label_template_path = label_template_path
        self.params_dict = params_dict

    def data_postprocess(self, model_outputs, filenames):
        """
        函数头不可更改，函数体自定义
        功能: 接收单个batch的模型输出预测值, 将其转化成指定的JSON数据格式. 主要用于模型感知阶段.
        model_outputs: 模型输出的单个batch的预测值, 亦即用户自定义的Networks.forward()方法的返回值.
        filenames: 单个batch中每帧样本的原始文件名,亦即用户自定义的data_dataset.py中Dataset. __getitem__()方法的返回值.
        return: 指定格式的预测结果,格式如下:
                [prediction of sample1, prediction of sample2 ...] 不同帧样本的结果以列表形式组织
                 prediction of sample1 --> 某一帧样本的结果以JSON格式组织(亦即Python中的dict字典数据类型);
                                           具体每帧样本标签应包含的信息,需结合数据管理平台获取的《label.json》和《每帧标签格式说明.json》
        注意：务必保证返回的预测结果和文件名称(列表形式)相对应.
        """
        predictions = []

        return predictions, filenames
