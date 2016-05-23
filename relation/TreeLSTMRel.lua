--[[

  Semantic relatedness prediction using Tree-LSTMs.

--]]

local TreeLSTMRel = torch.class('treelstm.TreeLSTMRel')

function TreeLSTMRel:__init(config)
  self.mem_dim       = config.mem_dim       or 150
  self.learning_rate = config.learning_rate or 0.01
  self.emb_learning_rate = config.emb_learning_rate or 0.01
  self.batch_size    = config.batch_size    or 25
  self.reg           = config.reg           or 1e-4
  self.structure     = config.structure     or 'dependency' -- {dependency, constituency}
  
  -- number of relation class
  self.num_classes   = config.relation_num


  -- word embedding
  self.emb_dim = config.emb_vecs:size(2)
  self.emb = nn.LookupTable(config.emb_vecs:size(1), self.emb_dim)
  self.emb = self.emb:cuda()

  -- position embedding
  self.pos_emb_dim = 5
  self.pos_emb1 = nn.LookupTable(61, self.pos_emb_dim)
  self.pos_emb1 = self.pos_emb1:cuda()
  self.pos_emb2 = nn.LookupTable(61, self.pos_emb_dim)
  self.pos_emb2 = self.pos_emb2:cuda()
  self.emb.weight:copy(config.emb_vecs)

  

  -- optimizer configuration
  self.optim_state = {}--{ learningRate = self.learning_rate }
  self.optim_config = {rho = 0.95, eps = 1e-6}

  -- KL divergence optimization objective
  self.criterion = nn.ClassNLLCriterion():cuda()

  -- initialize tree-lstm model
  local treelstm_config = {
    in_dim = self.emb_dim + 2 * self.pos_emb_dim,
    mem_dim = self.mem_dim,
    gate_output = false,
  }
  
  if self.structure == 'dependency' then
    self.treelstm = treelstm.ChildSumTreeLSTM(treelstm_config)
  elseif self.structure == 'constituency' then
    self.treelstm = treelstm.BinaryTreeLSTM(treelstm_config)
  else
    error('invalid parse tree type: ' .. self.structure)
  end

  -- relation and attention model
  self.rel_module = self:new_rel_module()
  self.att_module = self:new_att_module()
  local modules = nn.Parallel()
    :add(self.treelstm)
    :add(self.rel_module)
    :add(self.att_module)
    :add(self.emb)
    :add(self.pos_emb1)
    :add(self.pos_emb2)
  print (modules)
  self.params, self.grad_params = modules:getParameters()
end

function TreeLSTMRel:new_rel_module()
  local input = nn.Identity()()
  local d1 = nn.Dropout()(input)
  local h1 = nn.Linear(self.mem_dim, self.num_classes)(nn.Tanh()(d1))
 -- local h1 =  nn.TemporalConvolution(self.mem_dim, 1, 1)(nn.Tanh()(d1))
  local output = nn.LogSoftMax()(nn.Reshape(self.num_classes)(h1))

  local rel_module = nn.gModule({input},{output})
  rel_module:cuda()
  return rel_module
end

function TreeLSTMRel:new_att_module()
  local input = nn.Identity()()
 -- local h1 = nn.TemporalConvolution(self.mem_dim, self.num_classes, 1)(input)
 -- local attention = nn.SoftMax()(nn.Transpose({1,2})(h1))
 -- local output = nn.MM()({attention, input}) 
  local output = nn.Max(1)(input)

  local att_module = nn.gModule({input},{output})
  att_module:cuda()
  return att_module
end

function TreeLSTMRel:find(tree, e1_idx, e2_idx)
  local ok = {false, false}
  res = nil
  if tree.num_children == 0 then
    if (tree.leaf_idx==e1_idx) then
      ok[1] = true
    end
    if (tree.leaf_idx==e2_idx) then
      ok[2] = true
    end
  end
  for i = 1, tree.num_children do
    local tmp = self:find(tree.children[i], e1_idx, e2_idx)
   -- print (tmp)
    if (tmp[1]) then
      ok[1] = true
    end
    if (tmp[2]) then
      ok[2] = true
    end
    if (tmp[3]~=nil) then
      res = tmp[3]
      return {ok[1], ok[2], res, "DFS"}
      --print (res.num_children)
    end
  end
  if (res==nil and ok[1] and ok[2]) then
    res = tree
  end
  tree.ok = ok
  print (ok)
  return {ok[1], ok[2], res, "DFS"}
end

function TreeLSTMRel:train(dataset)
  self.treelstm:training()
  self.rel_module:training()
  self.att_module:training()
  local indices = torch.randperm(dataset.size)
  local zeros = torch.zeros(self.mem_dim)
  for i = 1, dataset.size, self.batch_size do
    xlua.progress(i, dataset.size)
    local batch_size = math.min(i + self.batch_size - 1, dataset.size) - i + 1

    local feval = function(x)
      self.grad_params:zero()
      self.emb:zeroGradParameters()
      self.pos_emb1:zeroGradParameters()
      self.pos_emb2:zeroGradParameters()
      local loss = 0
      for j = 1, batch_size do
        local idx = indices[i + j - 1]
        local tree = dataset.trees[idx]
        local sent = dataset.sents[idx]
        local e1 = dataset.e1[idx]
        local e2 = dataset.e2[idx]

        local inputs = self.emb:forward(sent)
        local inputs1 = self.pos_emb1:forward(e1)
        local inputs2 = self.pos_emb2:forward(e2)
        inputs = torch.cat(inputs, torch.cat(inputs1, inputs2))

        -- get sentence representations
        local rep = self.treelstm:forward(tree, inputs)[2]

        -- -- get all representations

        local rep_tree = self:find(tree, 31-e1[1], 31-e2[1])[3]
        local tree_table = tree:depth_first_preorder()
        for k,v in ipairs (tree_table) do
           v.grad = zeros:cuda()
        end
        local hidden = torch.Tensor(rep_tree:size(), self.mem_dim):cuda()
        tree_table = rep_tree:depth_first_preorder()
        for k,v in ipairs (tree_table) do
           hidden[k] = v.state[2]
        end
        rep = self.att_module:forward(hidden)

        -- rep = rep_tree.state[2]
        --print (rep)
        --print (torch.sum(rep, 2))

        -- compute relatedness
        local output = self.rel_module:forward(rep)

        -- compute loss and backpropagate
        --print (output)
        local example_loss = self.criterion:forward(output, dataset.rel[idx])
        loss = loss + example_loss
        local relation_grad = self.criterion:backward(output, dataset.rel[idx])
        local rep_grad = self.rel_module:backward(rep, relation_grad)
        local hidden_grad = self.att_module:backward(hidden, rep_grad)
        for k,v in ipairs (tree_table) do
          v.grad = hidden_grad[k]
        end
        -- rep_tree.grad = rep_grad

        local input_grads = self.treelstm:backward(tree, inputs, {zeros:cuda(), zeros:cuda()})
        self.emb:backward(sent, input_grads[{{}, {1,self.emb_dim}}])
        self.pos_emb1:backward(e1, input_grads[{{}, {self.emb_dim+1, self.emb_dim+self.pos_emb_dim}}])
        self.pos_emb2:backward(e2, input_grads[{{}, {self.emb_dim+self.pos_emb_dim+1, self.emb_dim+2*self.pos_emb_dim}}])
        --printf("%d\t%f\n", j, example_loss)
      end

      loss = loss / batch_size
      self.grad_params:div(batch_size)
      --self.emb.gradWeight:div(batch_size)
      --self.emb:updateParameters(self.emb_learning_rate)
      -- self.pos_emb1.gradWeight:div(batch_size)
      -- self.pos_emb1:updateParameters(self.emb_learning_rate)
      -- self.pos_emb2.gradWeight:div(batch_size)
      -- self.pos_emb2:updateParameters(self.emb_learning_rate)

      -- regularization
      loss = loss + 0.5 * self.reg * self.params:norm() ^ 2
      self.grad_params:add(self.reg, self.params)
      --print (loss)
      return loss, self.grad_params
    end

    optim.adadelta(feval, self.params, self.optim_config, self.optim_state)
  end
  xlua.progress(dataset.size, dataset.size)
end

-- Predict the relation of a sentence.
function TreeLSTMRel:predict(tree, sent, e1, e2)
  local inputs = self.emb:forward(sent)
  local inputs1 = self.pos_emb1:forward(e1)
  local inputs2 = self.pos_emb2:forward(e2)
  inputs = torch.cat(inputs, torch.cat(inputs1, inputs2))
  local rep = self.treelstm:forward(tree, inputs)[2]

  local rep_tree = self:find(tree, 31-e1[1], 31-e2[1])[3]
  local hidden = torch.Tensor(rep_tree:size(), self.mem_dim):cuda()
  local tree_table = rep_tree:depth_first_preorder()
  for k,v in ipairs (tree_table) do
    hidden[k] = v.state[2]
  end
  rep = self.att_module:forward(hidden)
  -- rep = rep_tree.state[2]

  local output = self.rel_module:forward(rep):double()
  output = torch.Tensor(output)
  self.treelstm:clean(tree)
  local tmp = output[1]
  local res = 1
  for i = 2, self.num_classes do
      --print (output[i])
      if output[i] > tmp then
        tmp = output[i]
        res = i
      end
  end
  --print (output[1])
  return {res, output}
end

-- Produce relation predictions for each sentence in the dataset.
function TreeLSTMRel:predict_dataset(dataset)
  self.treelstm:evaluate()
  self.rel_module:evaluate()
  self.att_module:evaluate()
  local predictions = {}
  for i = 1, dataset.size do
    xlua.progress(i, dataset.size)
    local tree = dataset.trees[i]
    local sent = dataset.sents[i]
    local e1 = dataset.e1[i]
    local e2 = dataset.e2[i]
    predictions[i] = self:predict(tree, sent, e1, e2)
  end
  return predictions
end

function TreeLSTMRel:debug(tree, sent, e1, e2, vocab)
  print ("debug")
  local inputs = self.emb:forward(sent)
  local inputs1 = self.pos_emb1:forward(e1)
  local inputs2 = self.pos_emb2:forward(e2)
  inputs = torch.cat(inputs, torch.cat(inputs1, inputs2))
  local rep = self.treelstm:forward(tree, inputs)[2]

  local rep_tree = self:find(tree, 31-e1[1], 31-e2[1])[3]
  rep_tree:debug(sent, vocab)
end

function TreeLSTMRel:print_config()
  local num_params = self.params:size(1)
  local num_rel_params = self:new_rel_module():getParameters():size(1)
  printf('%-25s = %d\n',   'num params', num_params)
  printf('%-25s = %d\n',   'num compositional params', num_params - num_rel_params)
  printf('%-25s = %d\n',   'word vector dim', self.emb_dim)
  printf('%-25s = %d\n',   'position vector dim', self.pos_emb_dim)
  printf('%-25s = %d\n',   'Tree-LSTM memory dim', self.mem_dim)
  printf('%-25s = %.2e\n', 'regularization strength', self.reg)
  printf('%-25s = %d\n',   'minibatch size', self.batch_size)
  printf('%-25s = %.2e\n', 'learning rate', self.learning_rate)
  printf('%-25s = %.2e\n', 'word vector learning rate', self.emb_learning_rate)
  printf('%-25s = %s\n',   'parse tree type', self.structure)
end

--
-- Serialization
--

function TreeLSTMRel:save(path)
  local config = {
    batch_size    = self.batch_size,
    emb_vecs      = self.emb.weight:float(),
    learning_rate = self.learning_rate,
    emb_learning_rate = self.emb_learning_rate,
    mem_dim       = self.mem_dim,
    reg           = self.reg,
    structure     = self.structure,
  }

  torch.save(path, {
    params = self.params,
    config = config,
  })
end

function TreeLSTMRel.load(path)
  local state = torch.load(path)
  local model = treelstm.TreeLSTMSim.new(state.config)
  model.params:copy(state.params)
  return model
end
