--[[

  Training script for relation classification.

--]]
require 'cutorch'
require 'cunn'

require('..')


-- read command line arguments
local args = lapp [[
Training script for semantic relatedness prediction on the SICK dataset.
  -m,--model  (default dependency) Model architecture: [dependency, constituency, lstm, bilstm]
  -l,--layers (default 1)          Number of layers (ignored for Tree-LSTM)
  -d,--dim    (default 150)        LSTM memory dimension
  -e,--epochs (default 10)         Number of training epochs
  -t,--type (default mean)
]]

local model_name, model_class
if args.model == 'dependency' then
  model_name = 'Dependency Tree LSTM'
  model_class = treelstm.TreeLSTMRel
elseif args.model == 'constituency' then
  model_name = 'Constituency Tree LSTM'
  model_class = treelstm.TreeLSTMRel
else
  error("invalid model type\n")
end
-- elseif args.model == 'lstm' then
--   model_name = 'LSTM'
--   model_class = treelstm.LSTMSim
-- elseif args.model == 'bilstm' then
--   model_name = 'Bidirectional LSTM'
--   model_class = treelstm.LSTMSim
-- end
local model_structure = args.model
header(model_name .. ' for Relation Classification')

-- directory containing dataset files
local data_dir = 'data/rel-class/'

-- load vocab
local vocab = treelstm.Vocab(data_dir .. 'vocab-cased.txt')

-- load embeddings
print('loading word embeddings')
-- local emb_dir = 'data/glove/'
-- local emb_prefix = emb_dir .. 'glove.840B'
local emb_dir = 'data/word2vec/'
local emb_prefix = emb_dir .. 'vec'
local emb_vocab, emb_vecs = treelstm.read_embedding(emb_prefix .. '.vocab', emb_prefix .. '.50d.th')
local emb_dim = emb_vecs:size(2)

local tmp_vocab = treelstm.Vocab("data/empty_vocab.txt")
print (vocab.size)
print (emb_vocab.size)
for i = 1, vocab.size do
  local w = vocab:token(i)
  if emb_vocab:contains(w) then
    tmp_vocab:add(w)
  end
end
print (tmp_vocab.size)
vocab = tmp_vocab


-- use only vectors in vocabulary (not necessary, but gives faster training)
local num_unk = 0
local vecs = torch.Tensor(vocab.size + 1, emb_dim)
for i = 1, vocab.size do
  local w = vocab:token(i)
  if emb_vocab:contains(w) then
    vecs[i] = emb_vecs[emb_vocab:index(w)]
  else
    num_unk = num_unk + 1
    vecs[i]:uniform(-0.05, 0.05)
  end
end
print('unk count = ' .. num_unk)
emb_vocab = nil
emb_vecs = nil
collectgarbage()

-- load datasets
print('loading datasets')
local train_dir = data_dir .. 'train/'
local test_dir = data_dir .. 'test/'
local constituency = (args.model == 'constituency')
local file_relation = io.open(data_dir .. 'relation2id.txt', 'r')
local relation2id = {}
local id2relation = {}
local rel_num = 0
local line
  while true do
    line = file_relation:read()
    if line == nil then break end
    local seg = stringx.split(line)
    relation2id[seg[1]] = tonumber(seg[2]) + 1
    id2relation[tonumber(seg[2]) + 1] = seg[1]
    rel_num = rel_num + 1
  end
print (relation2id)
local train_dataset = treelstm.read_relation_dataset(train_dir, vocab, relation2id, constituency)
local test_dataset = treelstm.read_relation_dataset(test_dir, vocab, relation2id, constituency)
printf('num train = %d\n', train_dataset.size)
printf('num test  = %d\n', test_dataset.size)

vocab:add("unk")

-- initialize model
local model = model_class{
  emb_vecs   = vecs,
  structure  = model_structure,
  num_layers = args.layers,
  mem_dim    = args.dim,
  relation_num = rel_num
}

-- number of epochs to train
local num_epochs = args.epochs

-- print information
header('model configuration')
printf('max epochs = %d\n', num_epochs)
model:print_config()



function precision(x, y, flag)
  local tmp = 0.0
  local len = #x
  for i = 1, len  do
    if x[i][1]==y[i] then
      tmp = tmp + 1
    else
      if flag and i > 2700 then
        printf("%d\t%d\t%s\t%d\t%s\n", i, x[i][1], id2relation[x[i][1]], y[i], id2relation[y[i]])
        for j = 1, 19 do 
          printf("%d:%.2f\t", j, x[i][2][j])
        end
        printf("\n")
      end
    end
  end
  return tmp/len
end

-- train
-- local debug_id = 2702
-- local tree = test_dataset.trees[debug_id]
-- local sent = test_dataset.sents[debug_id]
-- local e1 = test_dataset.e1[debug_id]
-- local e2 = test_dataset.e2[debug_id]
-- model:debug(tree,sent,e1,e2, vocab)
local train_start = sys.clock()
 for i = 1, 1 do
   local test_predictions = model:predict_dataset(test_dataset)
   local test_score = precision(test_predictions, test_dataset.rel, true)
   printf('-- test score: %.4f\n', test_score)

  -- for j = 1, #test_predictions do
  --    print (test_predictions[j][2][1])
  -- end
 end
header('Training model')
for i = 1, num_epochs do
  local start = sys.clock()
  printf('-- epoch %d\n', i)
  model:train(train_dataset)
  printf('-- finished epoch in %.2fs\n', sys.clock() - start)
  local test_predictions = model:predict_dataset(test_dataset)
  local test_score = precision(test_predictions, test_dataset.rel, true)
  printf('-- test score: %.4f\n', test_score)


  local train_predictions = model:predict_dataset(train_dataset)
  local train_score = precision(train_predictions, train_dataset.rel, false)
  printf('-- train score: %.4f\n', train_score)
  for j = 1, 19 do 
    printf("%d:%.2f\t", j, train_predictions[1][2][j])
  end
  -- printf("\n")
  print (train_dataset.rel[1])
end
printf('finished training in %.2fs\n', sys.clock() - train_start)


