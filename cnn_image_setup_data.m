%% 这是一个将自己的数据转换为Imdb格式的程序
% *********这个函数里面有几点需要注意的是，类别总数需要视自己的数据集修改。
function imdb = cnn_image_setup_data(varargin)

opts.dataDir = fullfile('J:\yd\fintuning\self_design_lenet\data\sketch\images2500test') ;
opts.lite = false ;% lite简化
opts = vl_argparse(opts, varargin) ;

% ------------------------------------------------------------------------
%                                                  Load categories metadata
% -------------------------------------------------------------------------

metaPath = fullfile(opts.dataDir, 'classInd.txt') ;%获得类别对应文件

fprintf('using metadata %s\n', metaPath) ;%使用哪里的数据
tmp = importdata(metaPath);%导入txt里面的txt文件的数据
nCls = numel(tmp);%里面有多少行
%% 判断类别与设定的是否一样 10为样本的类别总数（自己的数据集需要修改）
if nCls ~= 250
  error('Wrong meta file %s',metaPath);
end
% 将名字分离出来
cats = cell(1,nCls);
for i=1:numel(tmp)
  t = strsplit(tmp{i});
  cats{i} = t{2};
end
% 数据集文件夹选择
imdb.classes.name = cats ;
imdb.imageDir.train = fullfile(opts.dataDir, '3channal_train') ;
imdb.imageDir.test = fullfile(opts.dataDir, '3channal_test') ;

%% -----------------------------------------------------------------
%                                              load image names and labels
% -------------------------------------------------------------------------

name = {};
labels = {} ;
imdb.images.sets = [] ;

fprintf('searching training images ...\n') ;
%% 导入训练类别标签
train_label_path = fullfile(opts.dataDir, 'train_label.txt') ;
train_label_temp = importdata(train_label_path);%导入训练数据的标签文件txt.
temp_l = train_label_temp.data;%获取里面的10个类的数字信息
for i=1:numel(temp_l)
    train_label{i} = temp_l(i);
end
if length(train_label) ~= length(dir(fullfile(imdb.imageDir.train, '*.png')))
    error('training data is not equal to its label!!!');%如果txt里面的标签数目与jpg图像的数目不同，报错
end

i = 1;
for d = dir(fullfile(imdb.imageDir.train, '*.png'))'%获取图片文件夹的jpg文件
    name{end+1} = d.name;%把dir里面的name赋值给name.
    labels{end+1} = train_label{i} ;%标签就是前面的标签
    if mod(numel(name), 10) == 0, fprintf('.') ; end
    if mod(numel(name), 500) == 0, fprintf('\n') ; end
    imdb.images.sets(end+1) = 1;%train
    i = i+1;
end
%%
fprintf('searching testing images ...\n') ;
%% 导入测试类别标签和训练的类别标签一个套路
test_label_path = fullfile(opts.dataDir, 'test_label.txt') ;
test_label_temp = importdata(test_label_path);
temp_l = test_label_temp.data;
for i=1:numel(temp_l)
    test_label{i} = temp_l(i);
end
if length(test_label) ~= length(dir(fullfile(imdb.imageDir.test, '*.png')))
    error('testing data is not equal to its label!!!');
end
i = 1;
for d = dir(fullfile(imdb.imageDir.test, '*.png'))'
    name{end+1} = d.name;
    labels{end+1} = test_label{i} ;
    if mod(numel(name), 10) == 0, fprintf('.') ; end
    if mod(numel(name), 500) == 0, fprintf('\n') ; end
    imdb.images.sets(end+1) = 3;%test
    i = i+1;
end
%%
labels = horzcat(labels{:}) ;
imdb.images.id = 1:numel(name) ;
imdb.images.name = name ;
imdb.images.label = labels ;
