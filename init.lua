require('torch')
require('nn')
require('nngraph')
require('optim')
require('xlua')
require('sys')
require('lfs')

treelstm = {}

include('util/read_data.lua')
include('util/Tree.lua')
include('util/Vocab.lua')
include('layers/CRowAddTable.lua')
include('models/LSTM.lua')
include('models/TreeLSTM.lua')
include('models/ChildSumTreeLSTM.lua')
include('models/BinaryTreeLSTM.lua')
include('relatedness/LSTMSim.lua')
include('relatedness/TreeLSTMSim.lua')
include('sentiment/LSTMSentiment.lua')
include('sentiment/TreeLSTMSentiment.lua')

include('relation/TreeLSTMRel.lua')

printf = utils.printf

-- global paths (modify if desired)
treelstm.data_dir        = 'data'
treelstm.models_dir      = 'trained_models'
treelstm.predictions_dir = 'predictions'

-- share module parameters
function share_params(cell, src)
  if torch.type(cell) == 'nn.gModule' then
    for i = 1, #cell.forwardnodes do
      local node = cell.forwardnodes[i]
      local type1 = torch.type(cell.data)
      local type2 = torch.type(src.forwardnodes[i].data.module)
      --printf("cell %s\n", type1)
      --printf("src %s\n", type2)
      if node.data.module then
        node.data.module:share(src.forwardnodes[i].data.module,
          'weight', 'bias', 'gradWeight', 'gradBias')
      end
    end
  elseif torch.isTypeOf(cell, 'nn.Module') then
    cell:share(src, 'weight', 'bias', 'gradWeight', 'gradBias')
  else
    error('parameters cannot be shared for this input')
  end
end

function header(s)
  print(string.rep('-', 80))
  print(s)
  print(string.rep('-', 80))
end
