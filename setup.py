from distutils.core import setup
# from setuptools import setup, find_packages
setup(
    name='nntrainer',
    version='0.2',

    py_modules=['nntrainer.trainer.trainer_advanced','nntrainer.trainer.train','nntrainer.trainer.valid','nntrainer.trainer.predict',
                'nntrainer.trainer.am','nntrainer.trainer.model_saveload','nntrainer.trainer.max_min_stat','nntrainer.trainer.optimizer',
                'nntrainer.config_utils.args_updater','nntrainer.config_utils.seed','nntrainer.config_utils.yaml_loader',
                'nntrainer.data_utils.dataset_split','nntrainer.data_utils.dataset_loader',
                'nntrainer.data_utils.one_hot','nntrainer.data_utils.random_generator',
                'nntrainer.logging_utils.logger','nntrainer.logging_utils.check_point','nntrainer.logging_utils.pic2video',
                'nntrainer.model_utils.trivial','nntrainer.model_utils.fc','nntrainer.model_utils.convbase',
                'nntrainer.model_utils.weight_init',
                'nntrainer.model_utils.resnet',
                'nntrainer.model_utils.anode.odesolver','nntrainer.model_utils.anode.adjoint',
                'nntrainer.model_utils.anode.scheme','nntrainer.model_utils.anode.time_stepper',
                'nntrainer.model_utils.anode.ode_block',
                'nntrainer.model_utils.attention.att_conv2d',
                'nntrainer.model_utils.attention.bam','nntrainer.model_utils.attention.cbam',
                'nntrainer.model_utils.attention.residual_attention','nntrainer.model_utils.attention.squeeze_extraction',
                'nntrainer.model_utils.view','nntrainer.model_utils.loss','nntrainer.model_utils.similarity',
                'nntrainer.model_utils.model_parser','nntrainer.model_utils.help',
                'nntrainer.simple_datasets.xor','nntrainer.simple_datasets.sin','nntrainer.simple_datasets.dim_reduce',
                'nntrainer.simple_datasets.one_hot', 'nntrainer.simple_datasets.random_generator',
                'nntrainer.simple_datasets.mnist',]

)