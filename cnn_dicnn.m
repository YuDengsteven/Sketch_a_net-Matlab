function [net, info] = cnn_dicnn(varargin)
%CNN_dicnn 展示了一个预训练的CNN模型在Imagenet上面。
addpath('J:\yd\MatconvNet以Gpu运行，加特征融合\Matconvnet +GPU\matconvnet-1.0-beta23\examples\imagenet');
run('J:\yd\MatconvNet以Gpu运行，加特征融合\Matconvnet +GPU\matconvnet-1.0-beta23\matlab\vl_setupnn.m') ;
%修改图片读入的文件夹的路径

opts.dataDir = fullfile('J:\yd\fintuning\self_design_lenet\3layers\data_featurefusion') ;%输入数据的路径
opts.expDir  = fullfile('J:\yd\fintuning\self_design_lenet\3layers\data_featurefusion\sketch_a_net') ;
%导入预训练的model。(也就是我们要finetune的网络)。
% % % % % % % opts.modelPath = fullfile('J:\yd\MatconvNet以Gpu运行，加特征融合\Matconvnet +GPU\models','imagenet-caffe-alex.mat');
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.numFetchThreads = 12 ;

opts.lite = false ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');

opts.train = struct() ;
opts.train.gpus = [1];%选择使用GPU
opts.train.batchSize = 256 ;
opts.train.numSubBatches = 12 ;
opts.train.learningRate = 1e-4 * [ones(1,10), 0.1*ones(1,5)];

opts = vl_argparse(opts, varargin) ;
if ~isfield(opts.train, 'gpus'), opts.train.gpus = [1]; end;

%--------------------------------------------------------------------
%                                                      Prepare Models
%--------------------------------------------------------------------
% % % % % net = load(opts.modelPath);
% 修改一下这个model
net=get_sketch_a_net();
net = prepareDINet(net,opts);
%----------------------------------------------------------------------
%                                                      Prepare Data
%----------------------------------------------------------------------
% 准备数据格式
if exist(opts.imdbPath,'file')
  imdb = load(opts.imdbPath) ;
else
  imdb = cnn_image_setup_data('dataDir', opts.dataDir, 'lite', opts.lite) ;
  mkdir(opts.expDir) ;
  save(opts.imdbPath, '-struct', 'imdb') ;
end

imdb.images.set = imdb.images.sets;

% 在网络中设置类别的信息
net.meta.classes.name = imdb.classes.name ;
net.meta.classes.description = imdb.classes.name ;

%% 求训练集的均值
imageStatsPath = fullfile(opts.expDir, 'imageStats.mat') ;
if exist(imageStatsPath)
  load(imageStatsPath, 'averageImage') ;
else
    averageImage = getImageStats(opts, net.meta, imdb) ;
    save(imageStatsPath, 'averageImage') ;
end

% % 用新的均值改变均值
net.meta.normalization.averageImage = averageImage;
% -------------------------------------------------------------------------
%                                                                     Learn
% -------------------------------------------------------------------------
% 索引训练集==1  和测试集==3
opts.train.train = find(imdb.images.set==1) ;
opts.train.val = find(imdb.images.set==3) ;
% 训练
[net, info] = cnn_train_dag(net, imdb, getBatchFn(opts, net.meta), ...
                      'expDir', opts.expDir, ...
                      opts.train) ;


% -------------------------------------------------------------------------
%                                                                    Deploy
% -------------------------------------------------------------------------
% 保存训练完的网络
net = cnn_imagenet_deploy(net) ;
modelPath = fullfile(opts.expDir, 'net-deployed.mat');

net_ = net.saveobj() ;
save(modelPath, '-struct', 'net_') ;
clear net_ ;

% -------------------------------------------------------------------------
function fn = getBatchFn(opts, meta)
% -------------------------------------------------------------------------
useGpu = numel(opts.train.gpus) > 0 ;%是否使用GPU

bopts.numThreads = opts.numFetchThreads ;%这是什么还没弄明白。
bopts.imageSize = meta.normalization.imageSize ;%图片的尺寸。
% bopts.border = meta.normalization.border ;%图像的边界。
% bopts.averageImage = []; 
bopts.averageImage = meta.normalization.averageImage ;
% bopts.rgbVariance = meta.augmentation.rgbVariance ;
% bopts.transformation = meta.augmentation.transformation ;

fn = @(x,y) getDagNNBatch(bopts,useGpu,x,y) ;

% -------------------------------------------------------------------------
function inputs = getDagNNBatch(opts, useGpu, imdb, batch)
% -------------------------------------------------------------------------
% 判断读入数据为训练还是测试
for i = 1:length(batch)
    if imdb.images.set(batch(i)) == 1 %1为训练索引文件夹
        images(i) = strcat([imdb.imageDir.train filesep] , imdb.images.name(batch(i)));
    else
        images(i) = strcat([imdb.imageDir.test filesep] , imdb.images.name(batch(i)));
    end
end;
isVal = ~isempty(batch) && imdb.images.set(batch(1)) ~= 1 ;
if ~isVal
  % training
  im = cnn_imagenet_get_batch(images, opts, ...
                              'prefetch', nargout == 0) ;
else
  % validation: disable data augmentation
  im = cnn_imagenet_get_batch(images, opts, ...
                              'prefetch', nargout == 0, ...
                              'transformation', 'none') ;
end

if nargout > 0
  if useGpu
    im = gpuArray(im) ;
  end
  labels = imdb.images.label(batch) ;
  inputs = {'input', im, 'label', labels} ;
end

% 求训练样本的均值
% -------------------------------------------------------------------------
function averageImage = getImageStats(opts, meta, imdb)
% -------------------------------------------------------------------------
train = find(imdb.images.set == 1) ;
batch = 1:length(train);
fn = getBatchFn(opts, meta) ;
train = train(1: 100: end);
avg = {};
for i = 1:length(train)
    temp = fn(imdb, batch(train(i):train(i)+99)) ;
    temp = temp{2};
    avg{end+1} = mean(temp, 4) ;
end

averageImage = mean(cat(4,avg{:}),4) ;
% 将GPU格式的转化为cpu格式的保存起来（如果有用GPU）
averageImage = gather(averageImage);

   