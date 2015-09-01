require 'nn'
require 'nngraph'

LSTM = require 'LSTM.lua'

-- 3-layer LSTM network (input and output have 3 dimensions)
network = {LSTM.create(3, 4), LSTM.create(4, 4), LSTM.create(4, 3)}

-- network input
local x = torch.randn(1, 3)
local previous_state = {
  {torch.zeros(1, 4), torch.zeros(1,4)},
  {torch.zeros(1, 4), torch.zeros(1,4)},
  {torch.zeros(1, 3), torch.zeros(1,3)}
}

-- network output
output = nil
next_state = {}

-- forward pass
local layer_input = {x, table.unpack(previous_state[1])}
for l = 1, #network do
  -- forward the input
  local layer_output = network[l]:forward(layer_input)
  -- save output state for next iteration
  table.insert(next_state, layer_output)
  -- extract hidden state from output
  local layer_h = layer_output[2]
  -- prepare next layer's input or set the output
  if l < #network then
    layer_input = {layer_h, table.unpack(previous_state[l + 1])}
  else
    output = layer_h
  end
end

print(next_state)
print(output)
