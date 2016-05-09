%main_bp
%最初的bp所有激活函数均为sigmoid函数
function train_bp
%定义层向量，即每层的神经元数据
layers=[2 3 2]; % 输入层2个，隐含层3个，输出层2个


% 初始化权重
W=[];
for i=1:length(layers)-1
    W{i}=rand(layers(i),layers(i+1));
end



%整理样本数据，这里直接用随机数生成一些数据
x=[rand(2,5)+2 rand(2,5)-2];
d=[zeros(2,5) ones(2,5)];


%训练

for i=1:199 % 训练100次，然后每次迭代后的误差打印出来
      % 正向计算
%      net_h=x(:,mod(i,10)+1)'*W{1};
%      y=sigmoid(net_h);
%      net_o=y*W{2};
%      o=sigmoid(net_o);
     [o,y]=forward(x(:,mod(i,10)+1)',W);
     
     % 算误差
     error=d(:,mod(i,10)+1)'-o;
     [i,error]
     % 算各层的传播误差 % 算权重修正值
     ita=0.5;% 学历率
     % 梯度下降法计算delta_W
     delta_o=(d(:,mod(i,10)+1)'-o).*o.*(1-o);
     delta_W{2}=ita*y'*delta_o;
 
     delta_h=W{2}*delta_o';
     delta_W{1}=ita*x(:,mod(i,10)+1)*delta_h';
     
     
     % 更新权值，完成一次学习
     W{1}=W{1}+delta_W{1};
     W{2}=W{2}+delta_W{2};
     
end


% 测试
x=[rand(2,5)+2 rand(2,5)-2];
[o,y]=forward(x',W)


end

function [o,y]=forward(x,W)
   for i=1:size(x,1)
     net_h(i,:)=x(i,:)*W{1};
     y(i,:)=sigmoid(net_h(i,:));
     net_o(i,:)=y(i,:)*W{2};
     o(i,:)=sigmoid(net_o(i,:));
   end
end
function y=sigmoid(x)
    y=1./(1+exp(-x));
end
