%% ֮���ǶԵ����Ԥѵ��model����һ�㴦������һ������
% ������һ����Ҫ�Ĳ���������������nCls�������Ƕ�������޸Ķ��١�
% -------------------------------------------------------------------------
function net = prepareDINet(net,opts)
% -------------------------------------------------------------------------

% �滻fc8�㡣
fc8l = cellfun(@(a) strcmp(a.name, 'fc8'), net.layers)==1;

%%  note: ����������������һ��Ҫ���Լ���������Ǻϣ�����Ϊ10�ࣩ
nCls = 250;
sizeW = size(net.layers{fc8l}.weights{1});
% ��Ȩ�س�ʼ��
if sizeW(4)~=nCls
  net.layers{fc8l}.weights = {zeros(sizeW(1),sizeW(2),sizeW(3),nCls,'single'), ...
    zeros(1, nCls, 'single')};
end

% change loss  ���һ��loss������ѵ��
net.layers{end} = struct('name','loss', 'type','softmaxloss') ;

% convert to dagnn dagnn���磬����Ҫ��������⼸�����ѵ��
net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true) ;

net.addLayer('top1err', dagnn.Loss('loss', 'classerror'), ...
    {'prediction','label'}, 'top1err') ;
net.addLayer('top5err', dagnn.Loss('loss', 'topkerror', ...
    'opts', {'topK',5}), ...
{'prediction','label'}, 'top5err') ;