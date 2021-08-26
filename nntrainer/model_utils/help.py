HELP_convbase='ConvBaseBlock and ResConvBlock\n' \
              'ConvBaseBlock: Basic convolution blocks\n' \
              'Params:\n' \
              '\tnchannels:\tlist of numbers of channels. len(nchannels)-1 = number of layers\n' \
              '\tks:\tkernel sizes. list or int.\n' \
              '\tpool:\tpooling kernel and stride in the end. int. pool=-1 means no pooling.\n' \
              '\tactivate:\tactivation. list or string. relu, tanh, sigmoid are now supported.\n' \
              '\tbn:\tbatch norm required\n' \
              '\tbn_track:\tbn_track in batch norm\n' \
              'ResConvBaseBlock: Residual convolution blocks\n\n' \
     \
              'Params:\n' \
              '\tinchannel:\tlist of number of input channels.\n' \
              '\treschannel:\tlist of numbers of middle channels. inchannel!=reschannel will mean an extra layer to adjust the number of channels.\n' \
              '\tnlayer:\tnumber of layers\n' \
              '\tks:\tkernel sizes. list or int.\n' \
              '\tpool:\tpooling kernel and stride in the end. int. pool=-1 means no pooling.\n' \
              '\tactivate:\tactivation. list or string. relu, tanh, sigmoid are now supported.\n' \
              '\tbn:\tbatch norm required\n' \
              '\tbn_track:\tbn_track in batch norm\n'

HELP_fc='FCBlock_v2\n' \
              'FCBlock_v2: Basic linear blocks\n' \
              'Params:\n' \
              '\tshapes:\tlist shapes. len(shapes)-1 = number of layers\n' \
              '\tbn:\tbatch norm required\n' \
              '\tbn_track:\tbn_track in batch norm\n' \
              '\tactivate:\tactivation. list or string. relu, tanh, sigmoid are now supported.\n'

HELP_loss='Do not Use\n'

HELP_model_parser='Factory, NetworkFromFactory, parse_model, CascadedModels\n' \
              'Factory: Model Factory\n' \
              'Methods:\n' \
              '\tregister_item(k,v): add an item to the factory. factory_dict[k]=v\n' \
              '\tregister_dict(d): add all items in d to the factory.\n' \
              '\tcreate(modules):\tCreate a list of modules from the list of module definition dictionaries.\n' \
              '\t\tEach entry in modules should contain a type as the key of the module and the rest as its parameters.\n' \
              '\t\te.g. [{"type":"fc","shape":[2,3,1]},{"type":"cat","dim":-1}] means creating an fc module with params (shape=[2,3,1]) and a cat module with params (dim=-1)\n' \
              '\t\tNote that a module has to be registered before used, or the factory will look in nn.{type}\n\n' \
     \
              'NetworkFromFactory: Cascade modules created by the factory as one module\n' \
              'Params:\n' \
              '\tfactory: A factory as defined above.\n' \
              '\tmodules:\tA list of modules to be created from.\n\n' \
     \
              'parse_model: Recursively create modules from file list(or list of lists)\n' \
              'Params:\n' \
              '\tfnames: File name list. List-type entries will be automatically integrated. The yaml should contain "type" to specify the factory name, and "modules" for module attributes.\n' \
              '\tfactory: A dictionary of factories like {"fm":feature_map_factory,"fc":fc_factory...}.\n' \
              '\tparams: A dictionary of additional parameters defined either outside the function or in the model definition yaml.\n\n' \
     \
              'CascadedModels: Cascaded modules directly from files\n' \
              'Params:\n' \
              '\tfnames: File name list. List-type entries will be automatically integrated.\n' \
              '\tfactory: A dictionary of factories like {"fm":feature_map_factory,"fc":fc_factory...}.\n'

HELP_similarity='Do not Use\n'

HELP_view='View and Cat\n' \
          'View: reshape\n' \
          'Params:\n' \
          '\tshape:\toutput shape. -2 for automatically filling position with batch size\n\n' \
     \
          'Cat: concat\n' \
          'Params:\n' \
          '\tdim:\tdimension to cancat\n'

def help(name):
    s=eval(f'HELP_{name}')
    print(s)