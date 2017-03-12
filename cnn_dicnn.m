function [net, info] = cnn_dicnn(varargin)
%CNN_dicnn չʾ��һ��Ԥѵ����CNNģ����Imagenet���档
addpath('J:\yd\MatconvNet��Gpu���У��������ں�\Matconvnet +GPU\matconvnet-1.0-beta23\examples\imagenet');
run('J:\yd\MatconvNet��Gpu���У��������ں�\Matconvnet +GPU\matconvnet-1.0-beta23\matlab\vl_setupnn.m') ;
%�޸�ͼƬ������ļ��е�·��

opts.dataDir = fullfile('J:\yd\fintuning\self_design_lenet\3layers\data_featurefusion') ;%�������ݵ�·��
opts.expDir  = fullfile('J:\yd\fintuning\self_design_lenet\3layers\data_featurefusion\sketch_a_net') ;
%����Ԥѵ����model��(Ҳ��������Ҫfinetune������)��
% % % % % % % opts.modelPath = fullfile('J:\yd\MatconvNet��Gpu���У��������ں�\Matconvnet +GPU\models','imagenet-caffe-alex.mat');
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.numFetchThreads = 12 ;

opts.lite = false ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');

opts.train = struct() ;
opts.train.gpus = [1];%ѡ��ʹ��GPU
opts.train.batchSize = 256 ;
opts.train.numSubBatches = 12 ;
opts.train.learningRate = 1e-4 * [ones(1,10), 0.1*ones(1,5)];

opts = vl_argparse(opts, varargin) ;
if ~isfield(opts.train, 'gpus'), opts.train.gpus = [1]; end;

%--------------------------------------------------------------------
%                                                      Prepare Models
%--------------------------------------------------------------------
% % % % % net = load(opts.modelPath);
% �޸�һ�����model
net=get_sketch_a_net();
net = prepareDINet(net,opts);
%----------------------------------------------------------------------
%                                                      Prepare Data
%----------------------------------------------------------------------
% ׼�����ݸ�ʽ
if exist(opts.imdbPath,'file')
  imdb = load(opts.imdbPath) ;
else
  imdb = cnn_image_setup_data('dataDir', opts.dataDir, 'lite', opts.lite) ;
  mkdir(opts.expDir) ;
  save(opts.imdbPath, '-struct', 'imdb') ;
end

imdb.images.set = imdb.images.sets;

% ������������������Ϣ
net.meta.classes.name = imdb.classes.name ;
net.meta.classes.description = imdb.classes.name ;

%% ��ѵ�����ľ�ֵ
imageStatsPath = fullfile(opts.expDir, 'imageStats.mat') ;
if exist(imageStatsPath)
  load(imageStatsPath, 'averageImage') ;
else
    averageImage = getImageStats(opts, net.meta, imdb) ;
    save(imageStatsPath, 'averageImage') ;
end

% % ���µľ�ֵ�ı��ֵ
net.meta.normalization.averageImage = averageImage;
% -------------------------------------------------------------------------
%                                                                     Learn
% -------------------------------------------------------------------------
% ����ѵ����==1  �Ͳ��Լ�==3
opts.train.train = find(imdb.images.set==1) ;
opts.train.val = find(imdb.images.set==3) ;
% ѵ��
[net, info] = cnn_train_dag(net, imdb, getBatchFn(opts, net.meta), ...
                      'expDir', opts.expDir, ...
                      opts.train) ;


% -------------------------------------------------------------------------
%                                                                    Deploy
% -------------------------------------------------------------------------
% ����ѵ���������
net = cnn_imagenet_deploy(net) ;
modelPath = fullfile(opts.expDir, 'net-deployed.mat');

net_ = net.saveobj() ;
save(modelPath, '-struct', 'net_') ;
clear net_ ;

% -------------------------------------------------------------------------
function fn = getBatchFn(opts, meta)
% -------------------------------------------------------------------------
useGpu = numel(opts.train.gpus) > 0 ;%�Ƿ�ʹ��GPU

bopts.numThreads = opts.numFetchThreads ;%����ʲô��ûŪ���ס�
bopts.imageSize = meta.normalization.imageSize ;%ͼƬ�ĳߴ硣
% bopts.border = meta.normalization.border ;%ͼ��ı߽硣
% bopts.averageImage = []; 
bopts.averageImage = meta.normalization.averageImage ;
% bopts.rgbVariance = meta.augmentation.rgbVariance ;
% bopts.transformation = meta.augmentation.transformation ;

fn = @(x,y) getDagNNBatch(bopts,useGpu,x,y) ;

% -------------------------------------------------------------------------
function inputs = getDagNNBatch(opts, useGpu, imdb, batch)
% -------------------------------------------------------------------------
% �ж϶�������Ϊѵ�����ǲ���
for i = 1:length(batch)
    if imdb.images.set(batch(i)) == 1 %1Ϊѵ�������ļ���
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

% ��ѵ�������ľ�ֵ
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
% ��GPU��ʽ��ת��Ϊcpu��ʽ�ı����������������GPU��
averageImage = gather(averageImage);

   