%% 之后是对导入的预训练model进行一点处理，建立一个函数
% 这里有一个重要的参数就是你的类别数nCls，还是是多少类就修改多少。
% -------------------------------------------------------------------------
function net = prepareDINet(net,opts)
% -------------------------------------------------------------------------

% 替换fc8层。
fc8l = cellfun(@(a) strcmp(a.name, 'fc8'), net.layers)==1;

%%  note: 下面这个是类别数，一定要和自己的类别数吻合（这里为10类）
nCls = 250;
sizeW = size(net.layers{fc8l}.weights{1});
% 将权重初始化
if sizeW(4)~=nCls
  net.layers{fc8l}.weights = {zeros(sizeW(1),sizeW(2),sizeW(3),nCls,'single'), ...
    zeros(1, nCls, 'single')};
end

% change loss  添加一个loss层用于训练
net.layers{end} = struct('name','loss', 'type','softmaxloss') ;

% convert to dagnn dagnn网络，还需要添加下面这几层才能训练
net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true) ;

net.addLayer('top1err', dagnn.Loss('loss', 'classerror'), ...
    {'prediction','label'}, 'top1err') ;
net.addLayer('top5err', dagnn.Loss('loss', 'topkerror', ...
    'opts', {'topK',5}), ...
{'prediction','label'}, 'top5err') ;