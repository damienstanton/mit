// this one got 63mph
// experimental implementation of [Yu, Palefsky-Smith, Bedi, 2016]

//<![CDATA[

// a few things don't have var in front of them - they update already existing variables the game needs
lanesSide = 3;
patchesAhead = 50;
patchesBehind = 10;
trainIterations = 10000;

var num_inputs = (lanesSide * 2 + 1) * (patchesAhead + patchesBehind);
var num_actions = 5;
var temporal_window = 5;
var network_size = num_inputs * temporal_window + num_actions * temporal_window + num_inputs;

var layer_defs = [];
layer_defs.push({
    type: 'input',
    out_sx: 1,
    out_sy: 1,
    out_depth: network_size
});
layer_defs.push({
    type:'conv',
    sx:3,
    pad:1,
    filters:32,
    stride:1,
    activation:'relu'
});
layer_defs.push({
    type:'conv',
    sx:3,
    pad:1,
    filters:64,
    stride:1,
    activation:'relu'
});
layer_defs.push({
    type:'conv',
    sx:3,
    pad:1,
    filters:64,
    stride:1,
    activation:'relu'
});
layer_defs.push({
    type: 'fc',
    num_neurons: num_actions
});
layer_defs.push({
    type: 'fc',
    num_neurons: num_actions
});

var tdtrainer_options = {
    learning_rate: 0.001,
    momentum: 0.0,
    batch_size: 128,
    l2_decay: 0.01
};

var opt = {};
opt.temporal_window = temporal_window;
opt.experience_size = 3000;
opt.start_learn_threshold = 500;
opt.gamma = 0.7;
opt.learning_steps_total = 10000;
opt.learning_steps_burnin = 1000;
opt.epsilon_min = 0.0;
opt.epsilon_test_time = 0.0;
opt.layer_defs = layer_defs;
opt.tdtrainer_options = tdtrainer_options;

brain = new deepqlearn.Brain(num_inputs, num_actions, opt);

learn = function (state, lastReward) {
    brain.backward(lastReward);
    var action = brain.forward(state);

    draw_net();
    draw_stats();

    return action;
}

//]]>
    