# from options.train_options import TrainOptions
# import os

# class BaseConfig(TrainOptions):
#     def print_params(self, prtf=print):
#         prtf("")
#         prtf("Parameters:")
#         for attr, value in sorted(vars(self).items()):
#             prtf("{}={}".format(attr.upper(), value))
#         prtf("")

#     def as_markdown(self):
#         """ Return configs as markdown format """
#         text = "|name|value|  \n|-|-|  \n"
#         for attr, value in sorted(vars(self).items()):
#             text += "|{}|{}|  \n".format(attr, value)

#         return text

#     def initialize(self):
#         super().initialize()
#         self.type = "train"
