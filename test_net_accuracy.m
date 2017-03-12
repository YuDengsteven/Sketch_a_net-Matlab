clc
clear
% 导入model
% net2 = dagnn.DagNN.loadobj(load('J:\yd\MatconvNet以Gpu运行，加特征融合\Matconvnet +GPU\models\imagenet-vgg-f.mat')) ;
net1 = dagnn.DagNN.loadobj(load('J:\yd\fintuning\self_design_lenet\data\sketch\images2500test\exp_alex_3RGB\epoch_140\net-deployed.mat')) ;
net1.mode = 'test' ;
% 导入准备数据
imdb = load('J:\yd\fintuning\self_design_lenet\data\sketch\images2500test\exp_alex_3RGB\imdb.mat') ;

opts.dataDir = fullfile('J:\yd\fintuning\self_design_lenet\data\sketch\images2500test') ;
opts.expDir  = fullfile('J:\yd\fintuning\self_design_lenet\data\sketch\images2500test\exp_alex_3RGB') ;
% 找到训练与测试集
opts.train.train = find(imdb.images.sets==1) ;
opts.train.val = find(imdb.images.sets==3) ;

for i = 1:length(opts.train.val)
    i
    index = opts.train.val(i);
    label = imdb.images.label(index);
    % 读取测试的样本
    im_ =  imread(fullfile(imdb.imageDir.test,imdb.images.name{index}));
    im_ = single(im_);
    im_ = imresize(im_, net1.meta.normalization.imageSize(1:2)) ;
    im_ = bsxfun(@minus, im_, net1.meta.normalization.averageImage) ;
    % 测试
    net1.eval({'input',im_}) ;
    scores = net1.vars(net1.getVarIndex('prob')).value ;
    scores = squeeze(gather(scores)) ;

    [bestScore, best] = max(scores) ;
    truth(i) = label;
    pre(i) = best; 
end
% 计算准确率
accurcy = length(find(pre==truth))/length(truth);
disp(['accurcy = ',num2str(accurcy*100),'%']);