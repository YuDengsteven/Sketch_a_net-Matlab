clc
clear
% ����model
% net2 = dagnn.DagNN.loadobj(load('J:\yd\MatconvNet��Gpu���У��������ں�\Matconvnet +GPU\models\imagenet-vgg-f.mat')) ;
net1 = dagnn.DagNN.loadobj(load('J:\yd\fintuning\self_design_lenet\data\sketch\images2500test\exp_alex_3RGB\epoch_140\net-deployed.mat')) ;
net1.mode = 'test' ;
% ����׼������
imdb = load('J:\yd\fintuning\self_design_lenet\data\sketch\images2500test\exp_alex_3RGB\imdb.mat') ;

opts.dataDir = fullfile('J:\yd\fintuning\self_design_lenet\data\sketch\images2500test') ;
opts.expDir  = fullfile('J:\yd\fintuning\self_design_lenet\data\sketch\images2500test\exp_alex_3RGB') ;
% �ҵ�ѵ������Լ�
opts.train.train = find(imdb.images.sets==1) ;
opts.train.val = find(imdb.images.sets==3) ;

for i = 1:length(opts.train.val)
    i
    index = opts.train.val(i);
    label = imdb.images.label(index);
    % ��ȡ���Ե�����
    im_ =  imread(fullfile(imdb.imageDir.test,imdb.images.name{index}));
    im_ = single(im_);
    im_ = imresize(im_, net1.meta.normalization.imageSize(1:2)) ;
    im_ = bsxfun(@minus, im_, net1.meta.normalization.averageImage) ;
    % ����
    net1.eval({'input',im_}) ;
    scores = net1.vars(net1.getVarIndex('prob')).value ;
    scores = squeeze(gather(scores)) ;

    [bestScore, best] = max(scores) ;
    truth(i) = label;
    pre(i) = best; 
end
% ����׼ȷ��
accurcy = length(find(pre==truth))/length(truth);
disp(['accurcy = ',num2str(accurcy*100),'%']);