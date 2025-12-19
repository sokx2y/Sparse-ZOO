# forwarddelta路径 改进部分

  支持一次 lowrank_forward 就可以得到两个 loss 来估计梯度，以达到节省 memory cost 的目的 

## 1. fake_quant_mx.py 文件

  内有我们需要走 forward_delta 路径时构造的几个 layer： diffLinear, QdiffLInear, diffLayerNorm, diffEmbedding
  每一个层都有 forward（支持 base 的 单一input-> 单一output），对应着除了 train 之外的正常路径
  同时，每一层还有 forward_delta（支持 base 以及 diff 的双重输入输出）， 为了支持一次 lowrank_forward 就可以得到两个 loss的算子实现

  其中，QdiffLinear的逻辑是： 
  $$
    \begin{aligned}
    y_1 &= w_1 x + b_1 \\
    y_2 &= (w_1 + dw)(x + dx) + b_2 \\
        &= y_1 + dw\cdot x + w_1\cdot dx + dw\cdot dx + db
    \end{aligned}
  $$

  其中，x，dx, w, dw 分别有相应的量化格式 

  构造 replace 函数，将 model 中的 nnlayer 替换为对应的 difflayer 格式。


## 2. modeling_roberta.py & models.py 文件
  
  若 apply_forward_delta == True 则 将对应的 Linear，Embeeding，LayerNorm 用 fake_quant_mx 中的 replace 函数替换为diff形式的layer
  并且从底层到顶层依次新开一条 forward_delta 路径，打通 roberta 模型整体的 forward_delta

  P.S. robertaconfig里加上了forward_delta对应的几个参数并在每一层中的__init__中进行了difflayers的替换，但是由于模型初始化时对应的lozotrainer还未加载，uv_provider 和 z_provider还未加载， 所以最终有效的步骤还是run_lozo中用 mx 的 replace 函数也即 QuantizeRobertaForLOZO 来实现layer层面的替换， config的修改是冗余的（只是删掉需要修改一系列的__init__, 所以先放着了）


## 3. LOZOtrainer.py 文件
  
  make_uv_provider & make_z_provider: 缓存下来lowrank扰动矩阵，后续传入相应的difflayer复用

  lowrank_zo_step: 仅需一次 zo_forward 以及两次参数扰动 
  ```python
        self.lowrank_zo_perturb_parameters(scaling_factor=1)
        loss1, loss2 = self.zo_forward(model, inputs, with_delta = True)
        self.projected_grad = ((loss1 - loss2) / (2 * self.args.zo_eps)).item() 
        self.lowrank_zo_perturb_parameters(scaling_factor=-1)
  ```

  zo_forward: 对应调用 model 的 forward_delta 得到两个 loss

## 4. run_lozo.py 文件
  
  apply_forward_delta 控制是否开启 forward_delta路径

## 5. lozo.sh 中的对应参数

  ```bash
     APPLY_FORWARD_DELTA=${APPLY_FORWARD_DELTA:-true}
     ENABLE_X=${ENABLE_X:-true}
     ENABLE_DIFFX=${ENABLE_DIFFX:-true}
     ENABLE_W=${ENABLE_W:-true}
     ENABLE_DIFFW=${ENABLE_DIFFW:-true}
     MX_A_ELEM_FORMAT=${MX_A_ELEM_FORMAT:-"fp8_e4m3"}
     MX_DIFFA_ELEM_FORMAT=${MX_DIFFA_ELEM_FORMAT:-"fp4_e2m1"}
     MX_W_ELEM_FORMAT=${MX_W_ELEM_FORMAT:-"fp4_e2m1"}
     MX_DIFFW_ELEM_FORMAT=${MX_DIFFW_ELEM_FORMAT:-"fp4_e2m1"}
  ```
  

  
  

