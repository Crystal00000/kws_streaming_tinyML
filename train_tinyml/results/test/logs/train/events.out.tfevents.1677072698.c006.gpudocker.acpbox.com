       ?K"	  ?N???Abrain.Event:2+s^?n?     (???	?L?N???A"??
j
input_1Placeholder*'
_output_shapes
:?	*
shape:?	*
dtype0
?
max_pooling2d/MaxPoolMaxPoolinput_1*
explicit_paddings
 *
data_formatNHWC*
T0*
strides
*
ksize
*
paddingVALID*'
_output_shapes
:?
?
.stream/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*%
valueB"            *
dtype0* 
_class
loc:@stream/kernel
?
,stream/kernel/Initializer/random_uniform/minConst*
valueB
 *   ?*
dtype0* 
_class
loc:@stream/kernel*
_output_shapes
: 
?
,stream/kernel/Initializer/random_uniform/maxConst*
valueB
 *   ?*
dtype0*
_output_shapes
: * 
_class
loc:@stream/kernel
?
6stream/kernel/Initializer/random_uniform/RandomUniformRandomUniform.stream/kernel/Initializer/random_uniform/shape*&
_output_shapes
:*
dtype0*
seed2 *
T0* 
_class
loc:@stream/kernel*

seed 
?
,stream/kernel/Initializer/random_uniform/subSub,stream/kernel/Initializer/random_uniform/max,stream/kernel/Initializer/random_uniform/min*
_output_shapes
: * 
_class
loc:@stream/kernel*
T0
?
,stream/kernel/Initializer/random_uniform/mulMul6stream/kernel/Initializer/random_uniform/RandomUniform,stream/kernel/Initializer/random_uniform/sub*&
_output_shapes
:*
T0* 
_class
loc:@stream/kernel
?
(stream/kernel/Initializer/random_uniformAddV2,stream/kernel/Initializer/random_uniform/mul,stream/kernel/Initializer/random_uniform/min*&
_output_shapes
:*
T0* 
_class
loc:@stream/kernel
?
stream/kernelVarHandleOp*
shape:* 
_class
loc:@stream/kernel*
_output_shapes
: *
	container *
allowed_devices
 *
dtype0*
shared_namestream/kernel
k
.stream/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpstream/kernel*
_output_shapes
: 
?
stream/kernel/AssignAssignVariableOpstream/kernel(stream/kernel/Initializer/random_uniform*
dtype0*
validate_shape( 
w
!stream/kernel/Read/ReadVariableOpReadVariableOpstream/kernel*
dtype0*&
_output_shapes
:
?
stream/bias/Initializer/zerosConst*
valueB*    *
_output_shapes
:*
_class
loc:@stream/bias*
dtype0
?
stream/biasVarHandleOp*
	container *
_class
loc:@stream/bias*
allowed_devices
 *
shared_namestream/bias*
shape:*
dtype0*
_output_shapes
: 
g
,stream/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpstream/bias*
_output_shapes
: 
u
stream/bias/AssignAssignVariableOpstream/biasstream/bias/Initializer/zeros*
dtype0*
validate_shape( 
g
stream/bias/Read/ReadVariableOpReadVariableOpstream/bias*
_output_shapes
:*
dtype0
y
#stream/conv2d/Conv2D/ReadVariableOpReadVariableOpstream/kernel*&
_output_shapes
:*
dtype0
?
stream/conv2d/Conv2DConv2Dmax_pooling2d/MaxPool#stream/conv2d/Conv2D/ReadVariableOp*&
_output_shapes
:{*
paddingVALID*
use_cudnn_on_gpu(*
	dilations
*
T0*
strides
*
data_formatNHWC*
explicit_paddings
 
l
$stream/conv2d/BiasAdd/ReadVariableOpReadVariableOpstream/bias*
dtype0*
_output_shapes
:
?
stream/conv2d/BiasAddBiasAddstream/conv2d/Conv2D$stream/conv2d/BiasAdd/ReadVariableOp*
data_formatNHWC*
T0*&
_output_shapes
:{
?
*batch_normalization/gamma/Initializer/onesConst*
dtype0*
_output_shapes
:*
valueB*  ??*,
_class"
 loc:@batch_normalization/gamma
?
batch_normalization/gammaVarHandleOp*
_output_shapes
: **
shared_namebatch_normalization/gamma*
allowed_devices
 *
shape:*
	container *
dtype0*,
_class"
 loc:@batch_normalization/gamma
?
:batch_normalization/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization/gamma*
_output_shapes
: 
?
 batch_normalization/gamma/AssignAssignVariableOpbatch_normalization/gamma*batch_normalization/gamma/Initializer/ones*
dtype0*
validate_shape( 
?
-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes
:*
dtype0
?
*batch_normalization/beta/Initializer/zerosConst*
_output_shapes
:*
dtype0*
valueB*    *+
_class!
loc:@batch_normalization/beta
?
batch_normalization/betaVarHandleOp*
_output_shapes
: *+
_class!
loc:@batch_normalization/beta*
dtype0*
shape:*
allowed_devices
 *)
shared_namebatch_normalization/beta*
	container 
?
9batch_normalization/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization/beta*
_output_shapes
: 
?
batch_normalization/beta/AssignAssignVariableOpbatch_normalization/beta*batch_normalization/beta/Initializer/zeros*
dtype0*
validate_shape( 
?
,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
dtype0*
_output_shapes
:
?
1batch_normalization/moving_mean/Initializer/zerosConst*
_output_shapes
:*2
_class(
&$loc:@batch_normalization/moving_mean*
valueB*    *
dtype0
?
batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
	container *
shape:*
allowed_devices
 *0
shared_name!batch_normalization/moving_mean*2
_class(
&$loc:@batch_normalization/moving_mean
?
@batch_normalization/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization/moving_mean*
_output_shapes
: 
?
&batch_normalization/moving_mean/AssignAssignVariableOpbatch_normalization/moving_mean1batch_normalization/moving_mean/Initializer/zeros*
dtype0*
validate_shape( 
?
3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes
:*
dtype0
?
4batch_normalization/moving_variance/Initializer/onesConst*
valueB*  ??*6
_class,
*(loc:@batch_normalization/moving_variance*
_output_shapes
:*
dtype0
?
#batch_normalization/moving_varianceVarHandleOp*
shape:*6
_class,
*(loc:@batch_normalization/moving_variance*
	container *
dtype0*4
shared_name%#batch_normalization/moving_variance*
allowed_devices
 *
_output_shapes
: 
?
Dbatch_normalization/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp#batch_normalization/moving_variance*
_output_shapes
: 
?
*batch_normalization/moving_variance/AssignAssignVariableOp#batch_normalization/moving_variance4batch_normalization/moving_variance/Initializer/ones*
dtype0*
validate_shape( 
?
7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
dtype0*
_output_shapes
:
\
keras_learning_phase/inputConst*
dtype0
*
_output_shapes
: *
value	B
 Z 
|
keras_learning_phasePlaceholderWithDefaultkeras_learning_phase/input*
shape: *
dtype0
*
_output_shapes
: 
?
batch_normalization/condIfkeras_learning_phasebatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_variancestream/conv2d/BiasAdd*?
output_shapes.
,:{::: : : : : : : *
Tcond0
*
_lower_using_switch_merge(*@
_output_shapes.
,:{::: : : : : : : *&
_read_only_resource_inputs
*4
else_branch%R#
!batch_normalization_cond_false_40*
Tout
2
*3
then_branch$R"
 batch_normalization_cond_true_39*
Tin	
2
x
!batch_normalization/cond/IdentityIdentitybatch_normalization/cond*
T0*&
_output_shapes
:{
p
#batch_normalization/cond/Identity_1Identitybatch_normalization/cond:1*
_output_shapes
:*
T0
p
#batch_normalization/cond/Identity_2Identitybatch_normalization/cond:2*
_output_shapes
:*
T0
l
#batch_normalization/cond/Identity_3Identitybatch_normalization/cond:3*
T0*
_output_shapes
: 
l
#batch_normalization/cond/Identity_4Identitybatch_normalization/cond:4*
T0*
_output_shapes
: 
l
#batch_normalization/cond/Identity_5Identitybatch_normalization/cond:5*
_output_shapes
: *
T0
l
#batch_normalization/cond/Identity_6Identitybatch_normalization/cond:6*
T0*
_output_shapes
: 
l
#batch_normalization/cond/Identity_7Identitybatch_normalization/cond:7*
_output_shapes
: *
T0
l
#batch_normalization/cond/Identity_8Identitybatch_normalization/cond:8*
_output_shapes
: *
T0
l
#batch_normalization/cond/Identity_9Identitybatch_normalization/cond:9*
T0*
_output_shapes
: 
?
batch_normalization/cond_1StatelessIfkeras_learning_phase*
_output_shapes
: *7
else_branch(R&
$batch_normalization_cond_1_false_108* 
_read_only_resource_inputs
 *6
then_branch'R%
#batch_normalization_cond_1_true_107*	
Tin
 *
output_shapes
: *
Tcond0
*
Tout
2*
_lower_using_switch_merge(
l
#batch_normalization/cond_1/IdentityIdentitybatch_normalization/cond_1*
_output_shapes
: *
T0
?
)batch_normalization/AssignMovingAvg/sub/xConst*
_output_shapes
: *
valueB
 *  ??*2
_class(
&$loc:@batch_normalization/moving_mean*
dtype0
?
'batch_normalization/AssignMovingAvg/subSub)batch_normalization/AssignMovingAvg/sub/x#batch_normalization/cond_1/Identity*
_output_shapes
: *2
_class(
&$loc:@batch_normalization/moving_mean*
T0
?
2batch_normalization/AssignMovingAvg/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes
:*
dtype0
?
)batch_normalization/AssignMovingAvg/sub_1Sub2batch_normalization/AssignMovingAvg/ReadVariableOp#batch_normalization/cond/Identity_1*
T0*
_output_shapes
:*2
_class(
&$loc:@batch_normalization/moving_mean
?
'batch_normalization/AssignMovingAvg/mulMul)batch_normalization/AssignMovingAvg/sub_1'batch_normalization/AssignMovingAvg/sub*2
_class(
&$loc:@batch_normalization/moving_mean*
_output_shapes
:*
T0
?
7batch_normalization/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpbatch_normalization/moving_mean'batch_normalization/AssignMovingAvg/mul*2
_class(
&$loc:@batch_normalization/moving_mean*
dtype0
?
4batch_normalization/AssignMovingAvg/ReadVariableOp_1ReadVariableOpbatch_normalization/moving_mean8^batch_normalization/AssignMovingAvg/AssignSubVariableOp*2
_class(
&$loc:@batch_normalization/moving_mean*
dtype0*
_output_shapes
:
?
+batch_normalization/AssignMovingAvg_1/sub/xConst*6
_class,
*(loc:@batch_normalization/moving_variance*
_output_shapes
: *
valueB
 *  ??*
dtype0
?
)batch_normalization/AssignMovingAvg_1/subSub+batch_normalization/AssignMovingAvg_1/sub/x#batch_normalization/cond_1/Identity*
_output_shapes
: *
T0*6
_class,
*(loc:@batch_normalization/moving_variance
?
4batch_normalization/AssignMovingAvg_1/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes
:*
dtype0
?
+batch_normalization/AssignMovingAvg_1/sub_1Sub4batch_normalization/AssignMovingAvg_1/ReadVariableOp#batch_normalization/cond/Identity_2*
_output_shapes
:*
T0*6
_class,
*(loc:@batch_normalization/moving_variance
?
)batch_normalization/AssignMovingAvg_1/mulMul+batch_normalization/AssignMovingAvg_1/sub_1)batch_normalization/AssignMovingAvg_1/sub*6
_class,
*(loc:@batch_normalization/moving_variance*
_output_shapes
:*
T0
?
9batch_normalization/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp#batch_normalization/moving_variance)batch_normalization/AssignMovingAvg_1/mul*
dtype0*6
_class,
*(loc:@batch_normalization/moving_variance
?
6batch_normalization/AssignMovingAvg_1/ReadVariableOp_1ReadVariableOp#batch_normalization/moving_variance:^batch_normalization/AssignMovingAvg_1/AssignSubVariableOp*
_output_shapes
:*
dtype0*6
_class,
*(loc:@batch_normalization/moving_variance
k
activation/ReluRelu!batch_normalization/cond/Identity*
T0*&
_output_shapes
:{
?
:stream_1/depthwise_kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*%
valueB"            *
dtype0*,
_class"
 loc:@stream_1/depthwise_kernel
?
8stream_1/depthwise_kernel/Initializer/random_uniform/minConst*
dtype0*
valueB
 *?7?*,
_class"
 loc:@stream_1/depthwise_kernel*
_output_shapes
: 
?
8stream_1/depthwise_kernel/Initializer/random_uniform/maxConst*,
_class"
 loc:@stream_1/depthwise_kernel*
valueB
 *?7?*
_output_shapes
: *
dtype0
?
Bstream_1/depthwise_kernel/Initializer/random_uniform/RandomUniformRandomUniform:stream_1/depthwise_kernel/Initializer/random_uniform/shape*
T0*,
_class"
 loc:@stream_1/depthwise_kernel*

seed *
seed2 *
dtype0*&
_output_shapes
:
?
8stream_1/depthwise_kernel/Initializer/random_uniform/subSub8stream_1/depthwise_kernel/Initializer/random_uniform/max8stream_1/depthwise_kernel/Initializer/random_uniform/min*
_output_shapes
: *,
_class"
 loc:@stream_1/depthwise_kernel*
T0
?
8stream_1/depthwise_kernel/Initializer/random_uniform/mulMulBstream_1/depthwise_kernel/Initializer/random_uniform/RandomUniform8stream_1/depthwise_kernel/Initializer/random_uniform/sub*
T0*,
_class"
 loc:@stream_1/depthwise_kernel*&
_output_shapes
:
?
4stream_1/depthwise_kernel/Initializer/random_uniformAddV28stream_1/depthwise_kernel/Initializer/random_uniform/mul8stream_1/depthwise_kernel/Initializer/random_uniform/min*&
_output_shapes
:*
T0*,
_class"
 loc:@stream_1/depthwise_kernel
?
stream_1/depthwise_kernelVarHandleOp*
	container **
shared_namestream_1/depthwise_kernel*
shape:*,
_class"
 loc:@stream_1/depthwise_kernel*
dtype0*
_output_shapes
: *
allowed_devices
 
?
:stream_1/depthwise_kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpstream_1/depthwise_kernel*
_output_shapes
: 
?
 stream_1/depthwise_kernel/AssignAssignVariableOpstream_1/depthwise_kernel4stream_1/depthwise_kernel/Initializer/random_uniform*
dtype0*
validate_shape( 
?
-stream_1/depthwise_kernel/Read/ReadVariableOpReadVariableOpstream_1/depthwise_kernel*&
_output_shapes
:*
dtype0
?
stream_1/bias/Initializer/zerosConst*
_output_shapes
:* 
_class
loc:@stream_1/bias*
valueB*    *
dtype0
?
stream_1/biasVarHandleOp*
_output_shapes
: * 
_class
loc:@stream_1/bias*
dtype0*
	container *
shape:*
shared_namestream_1/bias*
allowed_devices
 
k
.stream_1/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpstream_1/bias*
_output_shapes
: 
{
stream_1/bias/AssignAssignVariableOpstream_1/biasstream_1/bias/Initializer/zeros*
dtype0*
validate_shape( 
k
!stream_1/bias/Read/ReadVariableOpReadVariableOpstream_1/bias*
_output_shapes
:*
dtype0
?
2stream_1/depthwise_conv2d/depthwise/ReadVariableOpReadVariableOpstream_1/depthwise_kernel*
dtype0*&
_output_shapes
:
?
)stream_1/depthwise_conv2d/depthwise/ShapeConst*
_output_shapes
:*%
valueB"            *
dtype0
?
1stream_1/depthwise_conv2d/depthwise/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
?
#stream_1/depthwise_conv2d/depthwiseDepthwiseConv2dNativeactivation/Relu2stream_1/depthwise_conv2d/depthwise/ReadVariableOp*
paddingVALID*
explicit_paddings
 *
data_formatNHWC*
	dilations
*
T0*&
_output_shapes
:<*
strides

z
0stream_1/depthwise_conv2d/BiasAdd/ReadVariableOpReadVariableOpstream_1/bias*
_output_shapes
:*
dtype0
?
!stream_1/depthwise_conv2d/BiasAddBiasAdd#stream_1/depthwise_conv2d/depthwise0stream_1/depthwise_conv2d/BiasAdd/ReadVariableOp*
data_formatNHWC*
T0*&
_output_shapes
:<
?
,batch_normalization_1/gamma/Initializer/onesConst*
_output_shapes
:*
valueB*  ??*
dtype0*.
_class$
" loc:@batch_normalization_1/gamma
?
batch_normalization_1/gammaVarHandleOp*
dtype0*
shape:*.
_class$
" loc:@batch_normalization_1/gamma*
	container *
_output_shapes
: *
allowed_devices
 *,
shared_namebatch_normalization_1/gamma
?
<batch_normalization_1/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_1/gamma*
_output_shapes
: 
?
"batch_normalization_1/gamma/AssignAssignVariableOpbatch_normalization_1/gamma,batch_normalization_1/gamma/Initializer/ones*
dtype0*
validate_shape( 
?
/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
dtype0*
_output_shapes
:
?
,batch_normalization_1/beta/Initializer/zerosConst*
valueB*    *
_output_shapes
:*-
_class#
!loc:@batch_normalization_1/beta*
dtype0
?
batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
	container *+
shared_namebatch_normalization_1/beta*
shape:*
allowed_devices
 *-
_class#
!loc:@batch_normalization_1/beta
?
;batch_normalization_1/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_1/beta*
_output_shapes
: 
?
!batch_normalization_1/beta/AssignAssignVariableOpbatch_normalization_1/beta,batch_normalization_1/beta/Initializer/zeros*
dtype0*
validate_shape( 
?
.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
dtype0*
_output_shapes
:
?
3batch_normalization_1/moving_mean/Initializer/zerosConst*
valueB*    *
dtype0*4
_class*
(&loc:@batch_normalization_1/moving_mean*
_output_shapes
:
?
!batch_normalization_1/moving_meanVarHandleOp*
shape:*
allowed_devices
 *
_output_shapes
: *
	container *
dtype0*4
_class*
(&loc:@batch_normalization_1/moving_mean*2
shared_name#!batch_normalization_1/moving_mean
?
Bbatch_normalization_1/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp!batch_normalization_1/moving_mean*
_output_shapes
: 
?
(batch_normalization_1/moving_mean/AssignAssignVariableOp!batch_normalization_1/moving_mean3batch_normalization_1/moving_mean/Initializer/zeros*
dtype0*
validate_shape( 
?
5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_output_shapes
:*
dtype0
?
6batch_normalization_1/moving_variance/Initializer/onesConst*
_output_shapes
:*
valueB*  ??*8
_class.
,*loc:@batch_normalization_1/moving_variance*
dtype0
?
%batch_normalization_1/moving_varianceVarHandleOp*
allowed_devices
 *
	container *
shape:*6
shared_name'%batch_normalization_1/moving_variance*
_output_shapes
: *
dtype0*8
_class.
,*loc:@batch_normalization_1/moving_variance
?
Fbatch_normalization_1/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp%batch_normalization_1/moving_variance*
_output_shapes
: 
?
,batch_normalization_1/moving_variance/AssignAssignVariableOp%batch_normalization_1/moving_variance6batch_normalization_1/moving_variance/Initializer/ones*
dtype0*
validate_shape( 
?
9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
_output_shapes
:*
dtype0
?
batch_normalization_1/condIfkeras_learning_phasebatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_variance!stream_1/depthwise_conv2d/BiasAdd*
_lower_using_switch_merge(*7
else_branch(R&
$batch_normalization_1_cond_false_165*6
then_branch'R%
#batch_normalization_1_cond_true_164*
Tout
2
*
Tcond0
*@
_output_shapes.
,:<::: : : : : : : *?
output_shapes.
,:<::: : : : : : : *&
_read_only_resource_inputs
*
Tin	
2
|
#batch_normalization_1/cond/IdentityIdentitybatch_normalization_1/cond*&
_output_shapes
:<*
T0
t
%batch_normalization_1/cond/Identity_1Identitybatch_normalization_1/cond:1*
T0*
_output_shapes
:
t
%batch_normalization_1/cond/Identity_2Identitybatch_normalization_1/cond:2*
T0*
_output_shapes
:
p
%batch_normalization_1/cond/Identity_3Identitybatch_normalization_1/cond:3*
T0*
_output_shapes
: 
p
%batch_normalization_1/cond/Identity_4Identitybatch_normalization_1/cond:4*
_output_shapes
: *
T0
p
%batch_normalization_1/cond/Identity_5Identitybatch_normalization_1/cond:5*
T0*
_output_shapes
: 
p
%batch_normalization_1/cond/Identity_6Identitybatch_normalization_1/cond:6*
_output_shapes
: *
T0
p
%batch_normalization_1/cond/Identity_7Identitybatch_normalization_1/cond:7*
T0*
_output_shapes
: 
p
%batch_normalization_1/cond/Identity_8Identitybatch_normalization_1/cond:8*
_output_shapes
: *
T0
p
%batch_normalization_1/cond/Identity_9Identitybatch_normalization_1/cond:9*
T0*
_output_shapes
: 
?
batch_normalization_1/cond_1StatelessIfkeras_learning_phase*
output_shapes
: *
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *
Tout
2*8
then_branch)R'
%batch_normalization_1_cond_1_true_232*9
else_branch*R(
&batch_normalization_1_cond_1_false_233*	
Tin
 *
Tcond0

p
%batch_normalization_1/cond_1/IdentityIdentitybatch_normalization_1/cond_1*
_output_shapes
: *
T0
?
+batch_normalization_1/AssignMovingAvg/sub/xConst*4
_class*
(&loc:@batch_normalization_1/moving_mean*
_output_shapes
: *
dtype0*
valueB
 *  ??
?
)batch_normalization_1/AssignMovingAvg/subSub+batch_normalization_1/AssignMovingAvg/sub/x%batch_normalization_1/cond_1/Identity*
T0*4
_class*
(&loc:@batch_normalization_1/moving_mean*
_output_shapes
: 
?
4batch_normalization_1/AssignMovingAvg/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_output_shapes
:*
dtype0
?
+batch_normalization_1/AssignMovingAvg/sub_1Sub4batch_normalization_1/AssignMovingAvg/ReadVariableOp%batch_normalization_1/cond/Identity_1*
_output_shapes
:*
T0*4
_class*
(&loc:@batch_normalization_1/moving_mean
?
)batch_normalization_1/AssignMovingAvg/mulMul+batch_normalization_1/AssignMovingAvg/sub_1)batch_normalization_1/AssignMovingAvg/sub*
T0*
_output_shapes
:*4
_class*
(&loc:@batch_normalization_1/moving_mean
?
9batch_normalization_1/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp!batch_normalization_1/moving_mean)batch_normalization_1/AssignMovingAvg/mul*4
_class*
(&loc:@batch_normalization_1/moving_mean*
dtype0
?
6batch_normalization_1/AssignMovingAvg/ReadVariableOp_1ReadVariableOp!batch_normalization_1/moving_mean:^batch_normalization_1/AssignMovingAvg/AssignSubVariableOp*4
_class*
(&loc:@batch_normalization_1/moving_mean*
_output_shapes
:*
dtype0
?
-batch_normalization_1/AssignMovingAvg_1/sub/xConst*
_output_shapes
: *8
_class.
,*loc:@batch_normalization_1/moving_variance*
valueB
 *  ??*
dtype0
?
+batch_normalization_1/AssignMovingAvg_1/subSub-batch_normalization_1/AssignMovingAvg_1/sub/x%batch_normalization_1/cond_1/Identity*8
_class.
,*loc:@batch_normalization_1/moving_variance*
_output_shapes
: *
T0
?
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
dtype0*
_output_shapes
:
?
-batch_normalization_1/AssignMovingAvg_1/sub_1Sub6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp%batch_normalization_1/cond/Identity_2*8
_class.
,*loc:@batch_normalization_1/moving_variance*
_output_shapes
:*
T0
?
+batch_normalization_1/AssignMovingAvg_1/mulMul-batch_normalization_1/AssignMovingAvg_1/sub_1+batch_normalization_1/AssignMovingAvg_1/sub*
T0*8
_class.
,*loc:@batch_normalization_1/moving_variance*
_output_shapes
:
?
;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp%batch_normalization_1/moving_variance+batch_normalization_1/AssignMovingAvg_1/mul*8
_class.
,*loc:@batch_normalization_1/moving_variance*
dtype0
?
8batch_normalization_1/AssignMovingAvg_1/ReadVariableOp_1ReadVariableOp%batch_normalization_1/moving_variance<^batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOp*
dtype0*
_output_shapes
:*8
_class.
,*loc:@batch_normalization_1/moving_variance
o
activation_1/ReluRelu#batch_normalization_1/cond/Identity*
T0*&
_output_shapes
:<
?
:stream_2/depthwise_kernel/Initializer/random_uniform/shapeConst*,
_class"
 loc:@stream_2/depthwise_kernel*
dtype0*%
valueB"            *
_output_shapes
:
?
8stream_2/depthwise_kernel/Initializer/random_uniform/minConst*
_output_shapes
: *
valueB
 *?7?*,
_class"
 loc:@stream_2/depthwise_kernel*
dtype0
?
8stream_2/depthwise_kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *,
_class"
 loc:@stream_2/depthwise_kernel*
valueB
 *?7?*
dtype0
?
Bstream_2/depthwise_kernel/Initializer/random_uniform/RandomUniformRandomUniform:stream_2/depthwise_kernel/Initializer/random_uniform/shape*&
_output_shapes
:*
seed2 *,
_class"
 loc:@stream_2/depthwise_kernel*

seed *
T0*
dtype0
?
8stream_2/depthwise_kernel/Initializer/random_uniform/subSub8stream_2/depthwise_kernel/Initializer/random_uniform/max8stream_2/depthwise_kernel/Initializer/random_uniform/min*
T0*
_output_shapes
: *,
_class"
 loc:@stream_2/depthwise_kernel
?
8stream_2/depthwise_kernel/Initializer/random_uniform/mulMulBstream_2/depthwise_kernel/Initializer/random_uniform/RandomUniform8stream_2/depthwise_kernel/Initializer/random_uniform/sub*&
_output_shapes
:*,
_class"
 loc:@stream_2/depthwise_kernel*
T0
?
4stream_2/depthwise_kernel/Initializer/random_uniformAddV28stream_2/depthwise_kernel/Initializer/random_uniform/mul8stream_2/depthwise_kernel/Initializer/random_uniform/min*&
_output_shapes
:*,
_class"
 loc:@stream_2/depthwise_kernel*
T0
?
stream_2/depthwise_kernelVarHandleOp*
dtype0*
	container *
allowed_devices
 **
shared_namestream_2/depthwise_kernel*
shape:*
_output_shapes
: *,
_class"
 loc:@stream_2/depthwise_kernel
?
:stream_2/depthwise_kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpstream_2/depthwise_kernel*
_output_shapes
: 
?
 stream_2/depthwise_kernel/AssignAssignVariableOpstream_2/depthwise_kernel4stream_2/depthwise_kernel/Initializer/random_uniform*
dtype0*
validate_shape( 
?
-stream_2/depthwise_kernel/Read/ReadVariableOpReadVariableOpstream_2/depthwise_kernel*&
_output_shapes
:*
dtype0
?
stream_2/bias/Initializer/zerosConst*
valueB*    *
dtype0*
_output_shapes
:* 
_class
loc:@stream_2/bias
?
stream_2/biasVarHandleOp*
	container *
dtype0*
_output_shapes
: *
shared_namestream_2/bias* 
_class
loc:@stream_2/bias*
allowed_devices
 *
shape:
k
.stream_2/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpstream_2/bias*
_output_shapes
: 
{
stream_2/bias/AssignAssignVariableOpstream_2/biasstream_2/bias/Initializer/zeros*
dtype0*
validate_shape( 
k
!stream_2/bias/Read/ReadVariableOpReadVariableOpstream_2/bias*
_output_shapes
:*
dtype0
?
4stream_2/depthwise_conv2d_1/depthwise/ReadVariableOpReadVariableOpstream_2/depthwise_kernel*&
_output_shapes
:*
dtype0
?
+stream_2/depthwise_conv2d_1/depthwise/ShapeConst*
dtype0*%
valueB"            *
_output_shapes
:
?
3stream_2/depthwise_conv2d_1/depthwise/dilation_rateConst*
valueB"      *
_output_shapes
:*
dtype0
?
%stream_2/depthwise_conv2d_1/depthwiseDepthwiseConv2dNativeactivation_1/Relu4stream_2/depthwise_conv2d_1/depthwise/ReadVariableOp*
explicit_paddings
 *
	dilations
*
strides
*
data_formatNHWC*
T0*
paddingVALID*&
_output_shapes
:9
|
2stream_2/depthwise_conv2d_1/BiasAdd/ReadVariableOpReadVariableOpstream_2/bias*
dtype0*
_output_shapes
:
?
#stream_2/depthwise_conv2d_1/BiasAddBiasAdd%stream_2/depthwise_conv2d_1/depthwise2stream_2/depthwise_conv2d_1/BiasAdd/ReadVariableOp*&
_output_shapes
:9*
data_formatNHWC*
T0
?
,batch_normalization_2/gamma/Initializer/onesConst*
_output_shapes
:*
valueB*  ??*
dtype0*.
_class$
" loc:@batch_normalization_2/gamma
?
batch_normalization_2/gammaVarHandleOp*.
_class$
" loc:@batch_normalization_2/gamma*
_output_shapes
: *
allowed_devices
 *
dtype0*
	container *,
shared_namebatch_normalization_2/gamma*
shape:
?
<batch_normalization_2/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_2/gamma*
_output_shapes
: 
?
"batch_normalization_2/gamma/AssignAssignVariableOpbatch_normalization_2/gamma,batch_normalization_2/gamma/Initializer/ones*
dtype0*
validate_shape( 
?
/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_2/gamma*
_output_shapes
:*
dtype0
?
,batch_normalization_2/beta/Initializer/zerosConst*
_output_shapes
:*-
_class#
!loc:@batch_normalization_2/beta*
dtype0*
valueB*    
?
batch_normalization_2/betaVarHandleOp*
allowed_devices
 *
dtype0*+
shared_namebatch_normalization_2/beta*
_output_shapes
: *-
_class#
!loc:@batch_normalization_2/beta*
shape:*
	container 
?
;batch_normalization_2/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_2/beta*
_output_shapes
: 
?
!batch_normalization_2/beta/AssignAssignVariableOpbatch_normalization_2/beta,batch_normalization_2/beta/Initializer/zeros*
dtype0*
validate_shape( 
?
.batch_normalization_2/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_2/beta*
_output_shapes
:*
dtype0
?
3batch_normalization_2/moving_mean/Initializer/zerosConst*4
_class*
(&loc:@batch_normalization_2/moving_mean*
dtype0*
_output_shapes
:*
valueB*    
?
!batch_normalization_2/moving_meanVarHandleOp*
	container *
allowed_devices
 *
_output_shapes
: *
dtype0*2
shared_name#!batch_normalization_2/moving_mean*
shape:*4
_class*
(&loc:@batch_normalization_2/moving_mean
?
Bbatch_normalization_2/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp!batch_normalization_2/moving_mean*
_output_shapes
: 
?
(batch_normalization_2/moving_mean/AssignAssignVariableOp!batch_normalization_2/moving_mean3batch_normalization_2/moving_mean/Initializer/zeros*
dtype0*
validate_shape( 
?
5batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*
dtype0*
_output_shapes
:
?
6batch_normalization_2/moving_variance/Initializer/onesConst*
dtype0*
_output_shapes
:*
valueB*  ??*8
_class.
,*loc:@batch_normalization_2/moving_variance
?
%batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *6
shared_name'%batch_normalization_2/moving_variance*
shape:*
	container *
allowed_devices
 *
dtype0*8
_class.
,*loc:@batch_normalization_2/moving_variance
?
Fbatch_normalization_2/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp%batch_normalization_2/moving_variance*
_output_shapes
: 
?
,batch_normalization_2/moving_variance/AssignAssignVariableOp%batch_normalization_2/moving_variance6batch_normalization_2/moving_variance/Initializer/ones*
dtype0*
validate_shape( 
?
9batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_2/moving_variance*
dtype0*
_output_shapes
:
?
batch_normalization_2/condIfkeras_learning_phasebatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_variance#stream_2/depthwise_conv2d_1/BiasAdd*
Tin	
2*&
_read_only_resource_inputs
*@
_output_shapes.
,:9::: : : : : : : *
Tcond0
*
_lower_using_switch_merge(*7
else_branch(R&
$batch_normalization_2_cond_false_290*?
output_shapes.
,:9::: : : : : : : *
Tout
2
*6
then_branch'R%
#batch_normalization_2_cond_true_289
|
#batch_normalization_2/cond/IdentityIdentitybatch_normalization_2/cond*
T0*&
_output_shapes
:9
t
%batch_normalization_2/cond/Identity_1Identitybatch_normalization_2/cond:1*
T0*
_output_shapes
:
t
%batch_normalization_2/cond/Identity_2Identitybatch_normalization_2/cond:2*
_output_shapes
:*
T0
p
%batch_normalization_2/cond/Identity_3Identitybatch_normalization_2/cond:3*
T0*
_output_shapes
: 
p
%batch_normalization_2/cond/Identity_4Identitybatch_normalization_2/cond:4*
T0*
_output_shapes
: 
p
%batch_normalization_2/cond/Identity_5Identitybatch_normalization_2/cond:5*
T0*
_output_shapes
: 
p
%batch_normalization_2/cond/Identity_6Identitybatch_normalization_2/cond:6*
T0*
_output_shapes
: 
p
%batch_normalization_2/cond/Identity_7Identitybatch_normalization_2/cond:7*
_output_shapes
: *
T0
p
%batch_normalization_2/cond/Identity_8Identitybatch_normalization_2/cond:8*
T0*
_output_shapes
: 
p
%batch_normalization_2/cond/Identity_9Identitybatch_normalization_2/cond:9*
_output_shapes
: *
T0
?
batch_normalization_2/cond_1StatelessIfkeras_learning_phase* 
_read_only_resource_inputs
 *
Tout
2*
output_shapes
: *
_output_shapes
: *8
then_branch)R'
%batch_normalization_2_cond_1_true_357*9
else_branch*R(
&batch_normalization_2_cond_1_false_358*
_lower_using_switch_merge(*	
Tin
 *
Tcond0

p
%batch_normalization_2/cond_1/IdentityIdentitybatch_normalization_2/cond_1*
_output_shapes
: *
T0
?
+batch_normalization_2/AssignMovingAvg/sub/xConst*
dtype0*
valueB
 *  ??*4
_class*
(&loc:@batch_normalization_2/moving_mean*
_output_shapes
: 
?
)batch_normalization_2/AssignMovingAvg/subSub+batch_normalization_2/AssignMovingAvg/sub/x%batch_normalization_2/cond_1/Identity*
_output_shapes
: *4
_class*
(&loc:@batch_normalization_2/moving_mean*
T0
?
4batch_normalization_2/AssignMovingAvg/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*
dtype0*
_output_shapes
:
?
+batch_normalization_2/AssignMovingAvg/sub_1Sub4batch_normalization_2/AssignMovingAvg/ReadVariableOp%batch_normalization_2/cond/Identity_1*4
_class*
(&loc:@batch_normalization_2/moving_mean*
_output_shapes
:*
T0
?
)batch_normalization_2/AssignMovingAvg/mulMul+batch_normalization_2/AssignMovingAvg/sub_1)batch_normalization_2/AssignMovingAvg/sub*
_output_shapes
:*4
_class*
(&loc:@batch_normalization_2/moving_mean*
T0
?
9batch_normalization_2/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp!batch_normalization_2/moving_mean)batch_normalization_2/AssignMovingAvg/mul*4
_class*
(&loc:@batch_normalization_2/moving_mean*
dtype0
?
6batch_normalization_2/AssignMovingAvg/ReadVariableOp_1ReadVariableOp!batch_normalization_2/moving_mean:^batch_normalization_2/AssignMovingAvg/AssignSubVariableOp*
dtype0*4
_class*
(&loc:@batch_normalization_2/moving_mean*
_output_shapes
:
?
-batch_normalization_2/AssignMovingAvg_1/sub/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ??*8
_class.
,*loc:@batch_normalization_2/moving_variance
?
+batch_normalization_2/AssignMovingAvg_1/subSub-batch_normalization_2/AssignMovingAvg_1/sub/x%batch_normalization_2/cond_1/Identity*
_output_shapes
: *8
_class.
,*loc:@batch_normalization_2/moving_variance*
T0
?
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpReadVariableOp%batch_normalization_2/moving_variance*
_output_shapes
:*
dtype0
?
-batch_normalization_2/AssignMovingAvg_1/sub_1Sub6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp%batch_normalization_2/cond/Identity_2*
T0*8
_class.
,*loc:@batch_normalization_2/moving_variance*
_output_shapes
:
?
+batch_normalization_2/AssignMovingAvg_1/mulMul-batch_normalization_2/AssignMovingAvg_1/sub_1+batch_normalization_2/AssignMovingAvg_1/sub*8
_class.
,*loc:@batch_normalization_2/moving_variance*
_output_shapes
:*
T0
?
;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp%batch_normalization_2/moving_variance+batch_normalization_2/AssignMovingAvg_1/mul*8
_class.
,*loc:@batch_normalization_2/moving_variance*
dtype0
?
8batch_normalization_2/AssignMovingAvg_1/ReadVariableOp_1ReadVariableOp%batch_normalization_2/moving_variance<^batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOp*
dtype0*8
_class.
,*loc:@batch_normalization_2/moving_variance*
_output_shapes
:
o
activation_2/ReluRelu#batch_normalization_2/cond/Identity*&
_output_shapes
:9*
T0
?
"stream_3/average_pooling2d/AvgPoolAvgPoolactivation_2/Relu*
data_formatNHWC*
strides
*
ksize
*
T0*
paddingVALID*&
_output_shapes
:
?
:stream_4/depthwise_kernel/Initializer/random_uniform/shapeConst*%
valueB"            *
_output_shapes
:*
dtype0*,
_class"
 loc:@stream_4/depthwise_kernel
?
8stream_4/depthwise_kernel/Initializer/random_uniform/minConst*
valueB
 *׳ݾ*
_output_shapes
: *,
_class"
 loc:@stream_4/depthwise_kernel*
dtype0
?
8stream_4/depthwise_kernel/Initializer/random_uniform/maxConst*
dtype0*
valueB
 *׳?>*
_output_shapes
: *,
_class"
 loc:@stream_4/depthwise_kernel
?
Bstream_4/depthwise_kernel/Initializer/random_uniform/RandomUniformRandomUniform:stream_4/depthwise_kernel/Initializer/random_uniform/shape*
seed2 *,
_class"
 loc:@stream_4/depthwise_kernel*

seed *&
_output_shapes
:*
dtype0*
T0
?
8stream_4/depthwise_kernel/Initializer/random_uniform/subSub8stream_4/depthwise_kernel/Initializer/random_uniform/max8stream_4/depthwise_kernel/Initializer/random_uniform/min*
_output_shapes
: *,
_class"
 loc:@stream_4/depthwise_kernel*
T0
?
8stream_4/depthwise_kernel/Initializer/random_uniform/mulMulBstream_4/depthwise_kernel/Initializer/random_uniform/RandomUniform8stream_4/depthwise_kernel/Initializer/random_uniform/sub*,
_class"
 loc:@stream_4/depthwise_kernel*
T0*&
_output_shapes
:
?
4stream_4/depthwise_kernel/Initializer/random_uniformAddV28stream_4/depthwise_kernel/Initializer/random_uniform/mul8stream_4/depthwise_kernel/Initializer/random_uniform/min*&
_output_shapes
:*
T0*,
_class"
 loc:@stream_4/depthwise_kernel
?
stream_4/depthwise_kernelVarHandleOp*
dtype0**
shared_namestream_4/depthwise_kernel*,
_class"
 loc:@stream_4/depthwise_kernel*
	container *
_output_shapes
: *
shape:*
allowed_devices
 
?
:stream_4/depthwise_kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpstream_4/depthwise_kernel*
_output_shapes
: 
?
 stream_4/depthwise_kernel/AssignAssignVariableOpstream_4/depthwise_kernel4stream_4/depthwise_kernel/Initializer/random_uniform*
dtype0*
validate_shape( 
?
-stream_4/depthwise_kernel/Read/ReadVariableOpReadVariableOpstream_4/depthwise_kernel*&
_output_shapes
:*
dtype0
?
stream_4/bias/Initializer/zerosConst*
_output_shapes
:*
valueB*    * 
_class
loc:@stream_4/bias*
dtype0
?
stream_4/biasVarHandleOp*
shape:*
dtype0*
_output_shapes
: *
shared_namestream_4/bias* 
_class
loc:@stream_4/bias*
allowed_devices
 *
	container 
k
.stream_4/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpstream_4/bias*
_output_shapes
: 
{
stream_4/bias/AssignAssignVariableOpstream_4/biasstream_4/bias/Initializer/zeros*
dtype0*
validate_shape( 
k
!stream_4/bias/Read/ReadVariableOpReadVariableOpstream_4/bias*
_output_shapes
:*
dtype0
?
4stream_4/depthwise_conv2d_2/depthwise/ReadVariableOpReadVariableOpstream_4/depthwise_kernel*&
_output_shapes
:*
dtype0
?
+stream_4/depthwise_conv2d_2/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            
?
3stream_4/depthwise_conv2d_2/depthwise/dilation_rateConst*
_output_shapes
:*
valueB"      *
dtype0
?
%stream_4/depthwise_conv2d_2/depthwiseDepthwiseConv2dNative"stream_3/average_pooling2d/AvgPool4stream_4/depthwise_conv2d_2/depthwise/ReadVariableOp*&
_output_shapes
:*
explicit_paddings
 *
strides
*
	dilations
*
data_formatNHWC*
paddingVALID*
T0
|
2stream_4/depthwise_conv2d_2/BiasAdd/ReadVariableOpReadVariableOpstream_4/bias*
dtype0*
_output_shapes
:
?
#stream_4/depthwise_conv2d_2/BiasAddBiasAdd%stream_4/depthwise_conv2d_2/depthwise2stream_4/depthwise_conv2d_2/BiasAdd/ReadVariableOp*
data_formatNHWC*&
_output_shapes
:*
T0
?
,batch_normalization_3/gamma/Initializer/onesConst*
valueB*  ??*.
_class$
" loc:@batch_normalization_3/gamma*
_output_shapes
:*
dtype0
?
batch_normalization_3/gammaVarHandleOp*,
shared_namebatch_normalization_3/gamma*
allowed_devices
 *
_output_shapes
: *
	container *.
_class$
" loc:@batch_normalization_3/gamma*
dtype0*
shape:
?
<batch_normalization_3/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_3/gamma*
_output_shapes
: 
?
"batch_normalization_3/gamma/AssignAssignVariableOpbatch_normalization_3/gamma,batch_normalization_3/gamma/Initializer/ones*
dtype0*
validate_shape( 
?
/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_3/gamma*
dtype0*
_output_shapes
:
?
,batch_normalization_3/beta/Initializer/zerosConst*
dtype0*
valueB*    *-
_class#
!loc:@batch_normalization_3/beta*
_output_shapes
:
?
batch_normalization_3/betaVarHandleOp*-
_class#
!loc:@batch_normalization_3/beta*
shape:*+
shared_namebatch_normalization_3/beta*
allowed_devices
 *
	container *
_output_shapes
: *
dtype0
?
;batch_normalization_3/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_3/beta*
_output_shapes
: 
?
!batch_normalization_3/beta/AssignAssignVariableOpbatch_normalization_3/beta,batch_normalization_3/beta/Initializer/zeros*
dtype0*
validate_shape( 
?
.batch_normalization_3/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_3/beta*
dtype0*
_output_shapes
:
?
3batch_normalization_3/moving_mean/Initializer/zerosConst*
_output_shapes
:*
dtype0*4
_class*
(&loc:@batch_normalization_3/moving_mean*
valueB*    
?
!batch_normalization_3/moving_meanVarHandleOp*
dtype0*
shape:*
_output_shapes
: *2
shared_name#!batch_normalization_3/moving_mean*4
_class*
(&loc:@batch_normalization_3/moving_mean*
	container *
allowed_devices
 
?
Bbatch_normalization_3/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp!batch_normalization_3/moving_mean*
_output_shapes
: 
?
(batch_normalization_3/moving_mean/AssignAssignVariableOp!batch_normalization_3/moving_mean3batch_normalization_3/moving_mean/Initializer/zeros*
dtype0*
validate_shape( 
?
5batch_normalization_3/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_3/moving_mean*
dtype0*
_output_shapes
:
?
6batch_normalization_3/moving_variance/Initializer/onesConst*
dtype0*
_output_shapes
:*8
_class.
,*loc:@batch_normalization_3/moving_variance*
valueB*  ??
?
%batch_normalization_3/moving_varianceVarHandleOp*
shape:*
dtype0*
_output_shapes
: *
allowed_devices
 *
	container *6
shared_name'%batch_normalization_3/moving_variance*8
_class.
,*loc:@batch_normalization_3/moving_variance
?
Fbatch_normalization_3/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp%batch_normalization_3/moving_variance*
_output_shapes
: 
?
,batch_normalization_3/moving_variance/AssignAssignVariableOp%batch_normalization_3/moving_variance6batch_normalization_3/moving_variance/Initializer/ones*
dtype0*
validate_shape( 
?
9batch_normalization_3/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_3/moving_variance*
dtype0*
_output_shapes
:
?
batch_normalization_3/condIfkeras_learning_phasebatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_variance#stream_4/depthwise_conv2d_2/BiasAdd*?
output_shapes.
,:::: : : : : : : *
Tin	
2*@
_output_shapes.
,:::: : : : : : : *&
_read_only_resource_inputs
*7
else_branch(R&
$batch_normalization_3_cond_false_416*
Tcond0
*
_lower_using_switch_merge(*
Tout
2
*6
then_branch'R%
#batch_normalization_3_cond_true_415
|
#batch_normalization_3/cond/IdentityIdentitybatch_normalization_3/cond*
T0*&
_output_shapes
:
t
%batch_normalization_3/cond/Identity_1Identitybatch_normalization_3/cond:1*
T0*
_output_shapes
:
t
%batch_normalization_3/cond/Identity_2Identitybatch_normalization_3/cond:2*
T0*
_output_shapes
:
p
%batch_normalization_3/cond/Identity_3Identitybatch_normalization_3/cond:3*
_output_shapes
: *
T0
p
%batch_normalization_3/cond/Identity_4Identitybatch_normalization_3/cond:4*
_output_shapes
: *
T0
p
%batch_normalization_3/cond/Identity_5Identitybatch_normalization_3/cond:5*
_output_shapes
: *
T0
p
%batch_normalization_3/cond/Identity_6Identitybatch_normalization_3/cond:6*
T0*
_output_shapes
: 
p
%batch_normalization_3/cond/Identity_7Identitybatch_normalization_3/cond:7*
T0*
_output_shapes
: 
p
%batch_normalization_3/cond/Identity_8Identitybatch_normalization_3/cond:8*
_output_shapes
: *
T0
p
%batch_normalization_3/cond/Identity_9Identitybatch_normalization_3/cond:9*
T0*
_output_shapes
: 
?
batch_normalization_3/cond_1StatelessIfkeras_learning_phase*8
then_branch)R'
%batch_normalization_3_cond_1_true_483* 
_read_only_resource_inputs
 *
Tcond0
*
output_shapes
: *
_lower_using_switch_merge(*	
Tin
 *
_output_shapes
: *
Tout
2*9
else_branch*R(
&batch_normalization_3_cond_1_false_484
p
%batch_normalization_3/cond_1/IdentityIdentitybatch_normalization_3/cond_1*
T0*
_output_shapes
: 
?
+batch_normalization_3/AssignMovingAvg/sub/xConst*
dtype0*4
_class*
(&loc:@batch_normalization_3/moving_mean*
valueB
 *  ??*
_output_shapes
: 
?
)batch_normalization_3/AssignMovingAvg/subSub+batch_normalization_3/AssignMovingAvg/sub/x%batch_normalization_3/cond_1/Identity*
T0*4
_class*
(&loc:@batch_normalization_3/moving_mean*
_output_shapes
: 
?
4batch_normalization_3/AssignMovingAvg/ReadVariableOpReadVariableOp!batch_normalization_3/moving_mean*
dtype0*
_output_shapes
:
?
+batch_normalization_3/AssignMovingAvg/sub_1Sub4batch_normalization_3/AssignMovingAvg/ReadVariableOp%batch_normalization_3/cond/Identity_1*
_output_shapes
:*
T0*4
_class*
(&loc:@batch_normalization_3/moving_mean
?
)batch_normalization_3/AssignMovingAvg/mulMul+batch_normalization_3/AssignMovingAvg/sub_1)batch_normalization_3/AssignMovingAvg/sub*
T0*
_output_shapes
:*4
_class*
(&loc:@batch_normalization_3/moving_mean
?
9batch_normalization_3/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp!batch_normalization_3/moving_mean)batch_normalization_3/AssignMovingAvg/mul*4
_class*
(&loc:@batch_normalization_3/moving_mean*
dtype0
?
6batch_normalization_3/AssignMovingAvg/ReadVariableOp_1ReadVariableOp!batch_normalization_3/moving_mean:^batch_normalization_3/AssignMovingAvg/AssignSubVariableOp*
_output_shapes
:*4
_class*
(&loc:@batch_normalization_3/moving_mean*
dtype0
?
-batch_normalization_3/AssignMovingAvg_1/sub/xConst*
dtype0*
valueB
 *  ??*8
_class.
,*loc:@batch_normalization_3/moving_variance*
_output_shapes
: 
?
+batch_normalization_3/AssignMovingAvg_1/subSub-batch_normalization_3/AssignMovingAvg_1/sub/x%batch_normalization_3/cond_1/Identity*
_output_shapes
: *8
_class.
,*loc:@batch_normalization_3/moving_variance*
T0
?
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOpReadVariableOp%batch_normalization_3/moving_variance*
dtype0*
_output_shapes
:
?
-batch_normalization_3/AssignMovingAvg_1/sub_1Sub6batch_normalization_3/AssignMovingAvg_1/ReadVariableOp%batch_normalization_3/cond/Identity_2*
T0*8
_class.
,*loc:@batch_normalization_3/moving_variance*
_output_shapes
:
?
+batch_normalization_3/AssignMovingAvg_1/mulMul-batch_normalization_3/AssignMovingAvg_1/sub_1+batch_normalization_3/AssignMovingAvg_1/sub*
_output_shapes
:*8
_class.
,*loc:@batch_normalization_3/moving_variance*
T0
?
;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp%batch_normalization_3/moving_variance+batch_normalization_3/AssignMovingAvg_1/mul*8
_class.
,*loc:@batch_normalization_3/moving_variance*
dtype0
?
8batch_normalization_3/AssignMovingAvg_1/ReadVariableOp_1ReadVariableOp%batch_normalization_3/moving_variance<^batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOp*
_output_shapes
:*
dtype0*8
_class.
,*loc:@batch_normalization_3/moving_variance
o
activation_3/ReluRelu#batch_normalization_3/cond/Identity*
T0*&
_output_shapes
:
?
:stream_5/depthwise_kernel/Initializer/random_uniform/shapeConst*%
valueB"            *,
_class"
 loc:@stream_5/depthwise_kernel*
dtype0*
_output_shapes
:
?
8stream_5/depthwise_kernel/Initializer/random_uniform/minConst*
valueB
 *b???*,
_class"
 loc:@stream_5/depthwise_kernel*
_output_shapes
: *
dtype0
?
8stream_5/depthwise_kernel/Initializer/random_uniform/maxConst*
dtype0*
valueB
 *b??>*
_output_shapes
: *,
_class"
 loc:@stream_5/depthwise_kernel
?
Bstream_5/depthwise_kernel/Initializer/random_uniform/RandomUniformRandomUniform:stream_5/depthwise_kernel/Initializer/random_uniform/shape*
seed2 *&
_output_shapes
:*,
_class"
 loc:@stream_5/depthwise_kernel*
dtype0*
T0*

seed 
?
8stream_5/depthwise_kernel/Initializer/random_uniform/subSub8stream_5/depthwise_kernel/Initializer/random_uniform/max8stream_5/depthwise_kernel/Initializer/random_uniform/min*,
_class"
 loc:@stream_5/depthwise_kernel*
_output_shapes
: *
T0
?
8stream_5/depthwise_kernel/Initializer/random_uniform/mulMulBstream_5/depthwise_kernel/Initializer/random_uniform/RandomUniform8stream_5/depthwise_kernel/Initializer/random_uniform/sub*&
_output_shapes
:*
T0*,
_class"
 loc:@stream_5/depthwise_kernel
?
4stream_5/depthwise_kernel/Initializer/random_uniformAddV28stream_5/depthwise_kernel/Initializer/random_uniform/mul8stream_5/depthwise_kernel/Initializer/random_uniform/min*&
_output_shapes
:*
T0*,
_class"
 loc:@stream_5/depthwise_kernel
?
stream_5/depthwise_kernelVarHandleOp*
shape:**
shared_namestream_5/depthwise_kernel*
allowed_devices
 *
dtype0*
	container *
_output_shapes
: *,
_class"
 loc:@stream_5/depthwise_kernel
?
:stream_5/depthwise_kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpstream_5/depthwise_kernel*
_output_shapes
: 
?
 stream_5/depthwise_kernel/AssignAssignVariableOpstream_5/depthwise_kernel4stream_5/depthwise_kernel/Initializer/random_uniform*
dtype0*
validate_shape( 
?
-stream_5/depthwise_kernel/Read/ReadVariableOpReadVariableOpstream_5/depthwise_kernel*&
_output_shapes
:*
dtype0
?
stream_5/bias/Initializer/zerosConst*
_output_shapes
:* 
_class
loc:@stream_5/bias*
dtype0*
valueB*    
?
stream_5/biasVarHandleOp*
dtype0*
_output_shapes
: *
allowed_devices
 *
shared_namestream_5/bias*
	container * 
_class
loc:@stream_5/bias*
shape:
k
.stream_5/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpstream_5/bias*
_output_shapes
: 
{
stream_5/bias/AssignAssignVariableOpstream_5/biasstream_5/bias/Initializer/zeros*
dtype0*
validate_shape( 
k
!stream_5/bias/Read/ReadVariableOpReadVariableOpstream_5/bias*
_output_shapes
:*
dtype0
?
4stream_5/depthwise_conv2d_3/depthwise/ReadVariableOpReadVariableOpstream_5/depthwise_kernel*
dtype0*&
_output_shapes
:
?
+stream_5/depthwise_conv2d_3/depthwise/ShapeConst*%
valueB"            *
dtype0*
_output_shapes
:
?
3stream_5/depthwise_conv2d_3/depthwise/dilation_rateConst*
valueB"      *
_output_shapes
:*
dtype0
?
%stream_5/depthwise_conv2d_3/depthwiseDepthwiseConv2dNativeactivation_3/Relu4stream_5/depthwise_conv2d_3/depthwise/ReadVariableOp*&
_output_shapes
:*
data_formatNHWC*
T0*
strides
*
explicit_paddings
 *
	dilations
*
paddingVALID
|
2stream_5/depthwise_conv2d_3/BiasAdd/ReadVariableOpReadVariableOpstream_5/bias*
dtype0*
_output_shapes
:
?
#stream_5/depthwise_conv2d_3/BiasAddBiasAdd%stream_5/depthwise_conv2d_3/depthwise2stream_5/depthwise_conv2d_3/BiasAdd/ReadVariableOp*&
_output_shapes
:*
data_formatNHWC*
T0
?
,batch_normalization_4/gamma/Initializer/onesConst*
dtype0*.
_class$
" loc:@batch_normalization_4/gamma*
_output_shapes
:*
valueB*  ??
?
batch_normalization_4/gammaVarHandleOp*
allowed_devices
 *,
shared_namebatch_normalization_4/gamma*
shape:*
_output_shapes
: *
dtype0*
	container *.
_class$
" loc:@batch_normalization_4/gamma
?
<batch_normalization_4/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_4/gamma*
_output_shapes
: 
?
"batch_normalization_4/gamma/AssignAssignVariableOpbatch_normalization_4/gamma,batch_normalization_4/gamma/Initializer/ones*
dtype0*
validate_shape( 
?
/batch_normalization_4/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_4/gamma*
dtype0*
_output_shapes
:
?
,batch_normalization_4/beta/Initializer/zerosConst*-
_class#
!loc:@batch_normalization_4/beta*
dtype0*
valueB*    *
_output_shapes
:
?
batch_normalization_4/betaVarHandleOp*+
shared_namebatch_normalization_4/beta*
allowed_devices
 *-
_class#
!loc:@batch_normalization_4/beta*
dtype0*
_output_shapes
: *
shape:*
	container 
?
;batch_normalization_4/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_4/beta*
_output_shapes
: 
?
!batch_normalization_4/beta/AssignAssignVariableOpbatch_normalization_4/beta,batch_normalization_4/beta/Initializer/zeros*
dtype0*
validate_shape( 
?
.batch_normalization_4/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_4/beta*
_output_shapes
:*
dtype0
?
3batch_normalization_4/moving_mean/Initializer/zerosConst*
dtype0*
_output_shapes
:*
valueB*    *4
_class*
(&loc:@batch_normalization_4/moving_mean
?
!batch_normalization_4/moving_meanVarHandleOp*
_output_shapes
: *4
_class*
(&loc:@batch_normalization_4/moving_mean*
allowed_devices
 *
shape:*
dtype0*
	container *2
shared_name#!batch_normalization_4/moving_mean
?
Bbatch_normalization_4/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp!batch_normalization_4/moving_mean*
_output_shapes
: 
?
(batch_normalization_4/moving_mean/AssignAssignVariableOp!batch_normalization_4/moving_mean3batch_normalization_4/moving_mean/Initializer/zeros*
dtype0*
validate_shape( 
?
5batch_normalization_4/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_4/moving_mean*
_output_shapes
:*
dtype0
?
6batch_normalization_4/moving_variance/Initializer/onesConst*
valueB*  ??*8
_class.
,*loc:@batch_normalization_4/moving_variance*
dtype0*
_output_shapes
:
?
%batch_normalization_4/moving_varianceVarHandleOp*
dtype0*
shape:*8
_class.
,*loc:@batch_normalization_4/moving_variance*6
shared_name'%batch_normalization_4/moving_variance*
	container *
_output_shapes
: *
allowed_devices
 
?
Fbatch_normalization_4/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp%batch_normalization_4/moving_variance*
_output_shapes
: 
?
,batch_normalization_4/moving_variance/AssignAssignVariableOp%batch_normalization_4/moving_variance6batch_normalization_4/moving_variance/Initializer/ones*
dtype0*
validate_shape( 
?
9batch_normalization_4/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_4/moving_variance*
dtype0*
_output_shapes
:
?
batch_normalization_4/condIfkeras_learning_phasebatch_normalization_4/gammabatch_normalization_4/beta!batch_normalization_4/moving_mean%batch_normalization_4/moving_variance#stream_5/depthwise_conv2d_3/BiasAdd*7
else_branch(R&
$batch_normalization_4_cond_false_541*
_lower_using_switch_merge(*&
_read_only_resource_inputs
*
Tcond0
*?
output_shapes.
,:::: : : : : : : *6
then_branch'R%
#batch_normalization_4_cond_true_540*
Tin	
2*@
_output_shapes.
,:::: : : : : : : *
Tout
2

|
#batch_normalization_4/cond/IdentityIdentitybatch_normalization_4/cond*&
_output_shapes
:*
T0
t
%batch_normalization_4/cond/Identity_1Identitybatch_normalization_4/cond:1*
T0*
_output_shapes
:
t
%batch_normalization_4/cond/Identity_2Identitybatch_normalization_4/cond:2*
_output_shapes
:*
T0
p
%batch_normalization_4/cond/Identity_3Identitybatch_normalization_4/cond:3*
T0*
_output_shapes
: 
p
%batch_normalization_4/cond/Identity_4Identitybatch_normalization_4/cond:4*
T0*
_output_shapes
: 
p
%batch_normalization_4/cond/Identity_5Identitybatch_normalization_4/cond:5*
T0*
_output_shapes
: 
p
%batch_normalization_4/cond/Identity_6Identitybatch_normalization_4/cond:6*
_output_shapes
: *
T0
p
%batch_normalization_4/cond/Identity_7Identitybatch_normalization_4/cond:7*
_output_shapes
: *
T0
p
%batch_normalization_4/cond/Identity_8Identitybatch_normalization_4/cond:8*
_output_shapes
: *
T0
p
%batch_normalization_4/cond/Identity_9Identitybatch_normalization_4/cond:9*
T0*
_output_shapes
: 
?
batch_normalization_4/cond_1StatelessIfkeras_learning_phase*	
Tin
 *8
then_branch)R'
%batch_normalization_4_cond_1_true_608*
output_shapes
: *
Tout
2*9
else_branch*R(
&batch_normalization_4_cond_1_false_609*
_output_shapes
: *
_lower_using_switch_merge(* 
_read_only_resource_inputs
 *
Tcond0

p
%batch_normalization_4/cond_1/IdentityIdentitybatch_normalization_4/cond_1*
T0*
_output_shapes
: 
?
+batch_normalization_4/AssignMovingAvg/sub/xConst*4
_class*
(&loc:@batch_normalization_4/moving_mean*
_output_shapes
: *
valueB
 *  ??*
dtype0
?
)batch_normalization_4/AssignMovingAvg/subSub+batch_normalization_4/AssignMovingAvg/sub/x%batch_normalization_4/cond_1/Identity*4
_class*
(&loc:@batch_normalization_4/moving_mean*
T0*
_output_shapes
: 
?
4batch_normalization_4/AssignMovingAvg/ReadVariableOpReadVariableOp!batch_normalization_4/moving_mean*
_output_shapes
:*
dtype0
?
+batch_normalization_4/AssignMovingAvg/sub_1Sub4batch_normalization_4/AssignMovingAvg/ReadVariableOp%batch_normalization_4/cond/Identity_1*
_output_shapes
:*4
_class*
(&loc:@batch_normalization_4/moving_mean*
T0
?
)batch_normalization_4/AssignMovingAvg/mulMul+batch_normalization_4/AssignMovingAvg/sub_1)batch_normalization_4/AssignMovingAvg/sub*
_output_shapes
:*
T0*4
_class*
(&loc:@batch_normalization_4/moving_mean
?
9batch_normalization_4/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp!batch_normalization_4/moving_mean)batch_normalization_4/AssignMovingAvg/mul*4
_class*
(&loc:@batch_normalization_4/moving_mean*
dtype0
?
6batch_normalization_4/AssignMovingAvg/ReadVariableOp_1ReadVariableOp!batch_normalization_4/moving_mean:^batch_normalization_4/AssignMovingAvg/AssignSubVariableOp*
dtype0*
_output_shapes
:*4
_class*
(&loc:@batch_normalization_4/moving_mean
?
-batch_normalization_4/AssignMovingAvg_1/sub/xConst*8
_class.
,*loc:@batch_normalization_4/moving_variance*
dtype0*
_output_shapes
: *
valueB
 *  ??
?
+batch_normalization_4/AssignMovingAvg_1/subSub-batch_normalization_4/AssignMovingAvg_1/sub/x%batch_normalization_4/cond_1/Identity*
_output_shapes
: *8
_class.
,*loc:@batch_normalization_4/moving_variance*
T0
?
6batch_normalization_4/AssignMovingAvg_1/ReadVariableOpReadVariableOp%batch_normalization_4/moving_variance*
_output_shapes
:*
dtype0
?
-batch_normalization_4/AssignMovingAvg_1/sub_1Sub6batch_normalization_4/AssignMovingAvg_1/ReadVariableOp%batch_normalization_4/cond/Identity_2*
_output_shapes
:*
T0*8
_class.
,*loc:@batch_normalization_4/moving_variance
?
+batch_normalization_4/AssignMovingAvg_1/mulMul-batch_normalization_4/AssignMovingAvg_1/sub_1+batch_normalization_4/AssignMovingAvg_1/sub*8
_class.
,*loc:@batch_normalization_4/moving_variance*
_output_shapes
:*
T0
?
;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp%batch_normalization_4/moving_variance+batch_normalization_4/AssignMovingAvg_1/mul*8
_class.
,*loc:@batch_normalization_4/moving_variance*
dtype0
?
8batch_normalization_4/AssignMovingAvg_1/ReadVariableOp_1ReadVariableOp%batch_normalization_4/moving_variance<^batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOp*
dtype0*8
_class.
,*loc:@batch_normalization_4/moving_variance*
_output_shapes
:
o
activation_4/ReluRelu#batch_normalization_4/cond/Identity*&
_output_shapes
:*
T0
?
dropout/condIfkeras_learning_phaseactivation_4/Relu*
Tcond0
*(
then_branchR
dropout_cond_true_629*
Tin
2*)
else_branchR
dropout_cond_false_630*4
_output_shapes"
 :: : : : : : : *
_lower_using_switch_merge(* 
_read_only_resource_inputs
 *
Tout

2*3
output_shapes"
 :: : : : : : : 
`
dropout/cond/IdentityIdentitydropout/cond*
T0*&
_output_shapes
:
T
dropout/cond/Identity_1Identitydropout/cond:1*
T0*
_output_shapes
: 
T
dropout/cond/Identity_2Identitydropout/cond:2*
T0*
_output_shapes
: 
T
dropout/cond/Identity_3Identitydropout/cond:3*
T0*
_output_shapes
: 
T
dropout/cond/Identity_4Identitydropout/cond:4*
_output_shapes
: *
T0
T
dropout/cond/Identity_5Identitydropout/cond:5*
_output_shapes
: *
T0
T
dropout/cond/Identity_6Identitydropout/cond:6*
_output_shapes
: *
T0
T
dropout/cond/Identity_7Identitydropout/cond:7*
T0*
_output_shapes
: 
g
stream_6/flatten/ConstConst*
valueB"????0   *
_output_shapes
:*
dtype0
?
stream_6/flatten/ReshapeReshapedropout/cond/Identitystream_6/flatten/Const*
Tshape0*
_output_shapes

:0*
T0
?
-dense/kernel/Initializer/random_uniform/shapeConst*
_class
loc:@dense/kernel*
_output_shapes
:*
valueB"0   
   *
dtype0
?
+dense/kernel/Initializer/random_uniform/minConst*
valueB
 *.???*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
: 
?
+dense/kernel/Initializer/random_uniform/maxConst*
valueB
 *.??>*
_output_shapes
: *
dtype0*
_class
loc:@dense/kernel
?
5dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform-dense/kernel/Initializer/random_uniform/shape*
_output_shapes

:0
*
_class
loc:@dense/kernel*

seed *
T0*
dtype0*
seed2 
?
+dense/kernel/Initializer/random_uniform/subSub+dense/kernel/Initializer/random_uniform/max+dense/kernel/Initializer/random_uniform/min*
_output_shapes
: *
_class
loc:@dense/kernel*
T0
?
+dense/kernel/Initializer/random_uniform/mulMul5dense/kernel/Initializer/random_uniform/RandomUniform+dense/kernel/Initializer/random_uniform/sub*
_output_shapes

:0
*
_class
loc:@dense/kernel*
T0
?
'dense/kernel/Initializer/random_uniformAddV2+dense/kernel/Initializer/random_uniform/mul+dense/kernel/Initializer/random_uniform/min*
_output_shapes

:0
*
_class
loc:@dense/kernel*
T0
?
dense/kernelVarHandleOp*
shared_namedense/kernel*
allowed_devices
 *
_class
loc:@dense/kernel*
	container *
shape
:0
*
_output_shapes
: *
dtype0
i
-dense/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense/kernel*
_output_shapes
: 
?
dense/kernel/AssignAssignVariableOpdense/kernel'dense/kernel/Initializer/random_uniform*
dtype0*
validate_shape( 
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
dtype0*
_output_shapes

:0

?
dense/bias/Initializer/zerosConst*
_output_shapes
:
*
valueB
*    *
dtype0*
_class
loc:@dense/bias
?

dense/biasVarHandleOp*
dtype0*
_output_shapes
: *
	container *
allowed_devices
 *
shape:
*
shared_name
dense/bias*
_class
loc:@dense/bias
e
+dense/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOp
dense/bias*
_output_shapes
: 
r
dense/bias/AssignAssignVariableOp
dense/biasdense/bias/Initializer/zeros*
dtype0*
validate_shape( 
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:
*
dtype0
h
dense/MatMul/ReadVariableOpReadVariableOpdense/kernel*
dtype0*
_output_shapes

:0

?
dense/MatMulMatMulstream_6/flatten/Reshapedense/MatMul/ReadVariableOp*
transpose_a( *
transpose_b( *
_output_shapes

:
*
T0
c
dense/BiasAdd/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:
*
dtype0
?
dense/BiasAddBiasAdddense/MatMuldense/BiasAdd/ReadVariableOp*
_output_shapes

:
*
data_formatNHWC*
T0
J

dense/ReluReludense/BiasAdd*
T0*
_output_shapes

:

?
/dense_1/kernel/Initializer/random_uniform/shapeConst*
valueB"
      *
dtype0*!
_class
loc:@dense_1/kernel*
_output_shapes
:
?
-dense_1/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *
dtype0*!
_class
loc:@dense_1/kernel*
valueB
 *?5?
?
-dense_1/kernel/Initializer/random_uniform/maxConst*
dtype0*
valueB
 *?5?*
_output_shapes
: *!
_class
loc:@dense_1/kernel
?
7dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_1/kernel/Initializer/random_uniform/shape*!
_class
loc:@dense_1/kernel*
T0*
dtype0*
_output_shapes

:
*
seed2 *

seed 
?
-dense_1/kernel/Initializer/random_uniform/subSub-dense_1/kernel/Initializer/random_uniform/max-dense_1/kernel/Initializer/random_uniform/min*
_output_shapes
: *!
_class
loc:@dense_1/kernel*
T0
?
-dense_1/kernel/Initializer/random_uniform/mulMul7dense_1/kernel/Initializer/random_uniform/RandomUniform-dense_1/kernel/Initializer/random_uniform/sub*
_output_shapes

:
*!
_class
loc:@dense_1/kernel*
T0
?
)dense_1/kernel/Initializer/random_uniformAddV2-dense_1/kernel/Initializer/random_uniform/mul-dense_1/kernel/Initializer/random_uniform/min*!
_class
loc:@dense_1/kernel*
T0*
_output_shapes

:

?
dense_1/kernelVarHandleOp*
shape
:
*
allowed_devices
 *!
_class
loc:@dense_1/kernel*
_output_shapes
: *
	container *
dtype0*
shared_namedense_1/kernel
m
/dense_1/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_1/kernel*
_output_shapes
: 
?
dense_1/kernel/AssignAssignVariableOpdense_1/kernel)dense_1/kernel/Initializer/random_uniform*
dtype0*
validate_shape( 
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

:
*
dtype0
?
dense_1/bias/Initializer/zerosConst*
valueB*    *
dtype0*
_class
loc:@dense_1/bias*
_output_shapes
:
?
dense_1/biasVarHandleOp*
_output_shapes
: *
	container *
_class
loc:@dense_1/bias*
shape:*
dtype0*
allowed_devices
 *
shared_namedense_1/bias
i
-dense_1/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_1/bias*
_output_shapes
: 
x
dense_1/bias/AssignAssignVariableOpdense_1/biasdense_1/bias/Initializer/zeros*
dtype0*
validate_shape( 
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:*
dtype0
l
dense_1/MatMul/ReadVariableOpReadVariableOpdense_1/kernel*
dtype0*
_output_shapes

:

?
dense_1/MatMulMatMul
dense/Reludense_1/MatMul/ReadVariableOp*
transpose_b( *
_output_shapes

:*
transpose_a( *
T0
g
dense_1/BiasAdd/ReadVariableOpReadVariableOpdense_1/bias*
dtype0*
_output_shapes
:
?
dense_1/BiasAddBiasAdddense_1/MatMuldense_1/BiasAdd/ReadVariableOp*
_output_shapes

:*
data_formatNHWC*
T0
T
dense_1/SoftmaxSoftmaxdense_1/BiasAdd*
T0*
_output_shapes

:ƪ
?
?
#batch_normalization_2_cond_true_2898
*readvariableop_batch_normalization_2_gamma:9
+readvariableop_1_batch_normalization_2_beta:O
Afusedbatchnormv3_readvariableop_batch_normalization_2_moving_mean:U
Gfusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance:8
4fusedbatchnormv3_stream_2_depthwise_conv2d_1_biasadd
fusedbatchnormv3
fusedbatchnormv3_0
fusedbatchnormv3_1
optionalfromvalue
optionalfromvalue_1
optionalfromvalue_2
optionalfromvalue_3
optionalfromvalue_4
optionalfromvalue_5
optionalfromvalue_6?u
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_2_gamma*
_output_shapes
:*
dtype0x
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_2_beta*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_2_moving_mean*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV34fusedbatchnormv3_stream_2_depthwise_conv2d_1_biasaddReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
epsilon%??'7*
U0*
T0*
is_training(*
exponential_avg_factor%  ??*
data_formatNHWC*B
_output_shapes0
.:9:::::?
OptionalFromValueOptionalFromValueReadVariableOp:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_1OptionalFromValueReadVariableOp_1:value:0*
_output_shapes
: *
Toutput_types
2:????
OptionalFromValue_2OptionalFromValue'FusedBatchNormV3/ReadVariableOp:value:0*
_output_shapes
: *
Toutput_types
2:????
OptionalFromValue_3OptionalFromValue)FusedBatchNormV3/ReadVariableOp_1:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_4OptionalFromValue"FusedBatchNormV3:reserve_space_1:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_5OptionalFromValue"FusedBatchNormV3:reserve_space_2:0*
_output_shapes
: *
Toutput_types
2:????
OptionalFromValue_6OptionalFromValue"FusedBatchNormV3:reserve_space_3:0*
_output_shapes
: *
Toutput_types
2:???"5
optionalfromvalue_2OptionalFromValue_2:optional:0"5
optionalfromvalue_3OptionalFromValue_3:optional:0"5
optionalfromvalue_4OptionalFromValue_4:optional:0"5
optionalfromvalue_5OptionalFromValue_5:optional:0"5
optionalfromvalue_6OptionalFromValue_6:optional:0"3
fusedbatchnormv3_0FusedBatchNormV3:batch_mean:0"7
fusedbatchnormv3_1!FusedBatchNormV3:batch_variance:0"1
optionalfromvalueOptionalFromValue:optional:0"(
fusedbatchnormv3FusedBatchNormV3:y:0"5
optionalfromvalue_1OptionalFromValue_1:optional:0*-
_input_shapes
: : : : :9:,(
&
_output_shapes
:9
?
2
%batch_normalization_4_cond_1_true_608	
constJ
ConstConst*
dtype0*
_output_shapes
: *
valueB
 *???="
constConst:output:0*
_input_shapes 
?
?
#batch_normalization_4_cond_true_5408
*readvariableop_batch_normalization_4_gamma:9
+readvariableop_1_batch_normalization_4_beta:O
Afusedbatchnormv3_readvariableop_batch_normalization_4_moving_mean:U
Gfusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance:8
4fusedbatchnormv3_stream_5_depthwise_conv2d_3_biasadd
fusedbatchnormv3
fusedbatchnormv3_0
fusedbatchnormv3_1
optionalfromvalue
optionalfromvalue_1
optionalfromvalue_2
optionalfromvalue_3
optionalfromvalue_4
optionalfromvalue_5
optionalfromvalue_6?u
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_4_gamma*
dtype0*
_output_shapes
:x
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_4_beta*
dtype0*
_output_shapes
:?
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_4_moving_mean*
dtype0*
_output_shapes
:?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV34fusedbatchnormv3_stream_5_depthwise_conv2d_3_biasaddReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
data_formatNHWC*
is_training(*
exponential_avg_factor%  ??*
U0*B
_output_shapes0
.::::::*
epsilon%??'7?
OptionalFromValueOptionalFromValueReadVariableOp:value:0*
_output_shapes
: *
Toutput_types
2:????
OptionalFromValue_1OptionalFromValueReadVariableOp_1:value:0*
_output_shapes
: *
Toutput_types
2:????
OptionalFromValue_2OptionalFromValue'FusedBatchNormV3/ReadVariableOp:value:0*
_output_shapes
: *
Toutput_types
2:????
OptionalFromValue_3OptionalFromValue)FusedBatchNormV3/ReadVariableOp_1:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_4OptionalFromValue"FusedBatchNormV3:reserve_space_1:0*
_output_shapes
: *
Toutput_types
2:????
OptionalFromValue_5OptionalFromValue"FusedBatchNormV3:reserve_space_2:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_6OptionalFromValue"FusedBatchNormV3:reserve_space_3:0*
Toutput_types
2*
_output_shapes
: :???"7
fusedbatchnormv3_1!FusedBatchNormV3:batch_variance:0"1
optionalfromvalueOptionalFromValue:optional:0"(
fusedbatchnormv3FusedBatchNormV3:y:0"5
optionalfromvalue_1OptionalFromValue_1:optional:0"5
optionalfromvalue_2OptionalFromValue_2:optional:0"5
optionalfromvalue_3OptionalFromValue_3:optional:0"5
optionalfromvalue_4OptionalFromValue_4:optional:0"5
optionalfromvalue_5OptionalFromValue_5:optional:0"5
optionalfromvalue_6OptionalFromValue_6:optional:0"3
fusedbatchnormv3_0FusedBatchNormV3:batch_mean:0*-
_input_shapes
: : : : ::,(
&
_output_shapes
:
?
?
$batch_normalization_3_cond_false_4168
*readvariableop_batch_normalization_3_gamma:9
+readvariableop_1_batch_normalization_3_beta:O
Afusedbatchnormv3_readvariableop_batch_normalization_3_moving_mean:U
Gfusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance:8
4fusedbatchnormv3_stream_4_depthwise_conv2d_2_biasadd
fusedbatchnormv3
fusedbatchnormv3_0
fusedbatchnormv3_1
optionalfromvalue
optionalfromvalue_1
optionalfromvalue_2
optionalfromvalue_3
optionalfromvalue_4
optionalfromvalue_5
optionalfromvalue_6?u
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_3_gamma*
dtype0*
_output_shapes
:x
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_3_beta*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_3_moving_mean*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV34fusedbatchnormv3_stream_4_depthwise_conv2d_2_biasaddReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
exponential_avg_factor%  ??*
data_formatNHWC*
U0*
T0*
epsilon%??'7*
is_training( *B
_output_shapes0
.::::::?
OptionalFromValueOptionalFromValueReadVariableOp:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_1OptionalFromValueReadVariableOp_1:value:0*
_output_shapes
: *
Toutput_types
2:????
OptionalFromValue_2OptionalFromValue'FusedBatchNormV3/ReadVariableOp:value:0*
_output_shapes
: *
Toutput_types
2:????
OptionalFromValue_3OptionalFromValue)FusedBatchNormV3/ReadVariableOp_1:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_4OptionalFromValue"FusedBatchNormV3:reserve_space_1:0*
_output_shapes
: *
Toutput_types
2:????
OptionalFromValue_5OptionalFromValue"FusedBatchNormV3:reserve_space_2:0*
_output_shapes
: *
Toutput_types
2:????
OptionalFromValue_6OptionalFromValue"FusedBatchNormV3:reserve_space_3:0*
Toutput_types
2*
_output_shapes
: :???"5
optionalfromvalue_6OptionalFromValue_6:optional:0"3
fusedbatchnormv3_0FusedBatchNormV3:batch_mean:0"7
fusedbatchnormv3_1!FusedBatchNormV3:batch_variance:0"1
optionalfromvalueOptionalFromValue:optional:0"(
fusedbatchnormv3FusedBatchNormV3:y:0"5
optionalfromvalue_1OptionalFromValue_1:optional:0"5
optionalfromvalue_2OptionalFromValue_2:optional:0"5
optionalfromvalue_3OptionalFromValue_3:optional:0"5
optionalfromvalue_4OptionalFromValue_4:optional:0"5
optionalfromvalue_5OptionalFromValue_5:optional:0*-
_input_shapes
: : : : ::,(
&
_output_shapes
:
?
3
&batch_normalization_1_cond_1_false_233	
constJ
ConstConst*
dtype0*
_output_shapes
: *
valueB
 *  ??"
constConst:output:0*
_input_shapes 
?
2
%batch_normalization_2_cond_1_true_357	
constJ
ConstConst*
valueB
 *???=*
_output_shapes
: *
dtype0"
constConst:output:0*
_input_shapes 
?
3
&batch_normalization_3_cond_1_false_484	
constJ
ConstConst*
_output_shapes
: *
valueB
 *  ??*
dtype0"
constConst:output:0*
_input_shapes 
?
?
 batch_normalization_cond_true_396
(readvariableop_batch_normalization_gamma:7
)readvariableop_1_batch_normalization_beta:M
?fusedbatchnormv3_readvariableop_batch_normalization_moving_mean:S
Efusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance:*
&fusedbatchnormv3_stream_conv2d_biasadd
fusedbatchnormv3
fusedbatchnormv3_0
fusedbatchnormv3_1
optionalfromvalue
optionalfromvalue_1
optionalfromvalue_2
optionalfromvalue_3
optionalfromvalue_4
optionalfromvalue_5
optionalfromvalue_6?s
ReadVariableOpReadVariableOp(readvariableop_batch_normalization_gamma*
_output_shapes
:*
dtype0v
ReadVariableOp_1ReadVariableOp)readvariableop_1_batch_normalization_beta*
dtype0*
_output_shapes
:?
FusedBatchNormV3/ReadVariableOpReadVariableOp?fusedbatchnormv3_readvariableop_batch_normalization_moving_mean*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpEfusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance*
dtype0*
_output_shapes
:?
FusedBatchNormV3FusedBatchNormV3&fusedbatchnormv3_stream_conv2d_biasaddReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
U0*
T0*
is_training(*B
_output_shapes0
.:{:::::*
data_formatNHWC*
epsilon%??'7*
exponential_avg_factor%  ???
OptionalFromValueOptionalFromValueReadVariableOp:value:0*
_output_shapes
: *
Toutput_types
2:????
OptionalFromValue_1OptionalFromValueReadVariableOp_1:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_2OptionalFromValue'FusedBatchNormV3/ReadVariableOp:value:0*
_output_shapes
: *
Toutput_types
2:????
OptionalFromValue_3OptionalFromValue)FusedBatchNormV3/ReadVariableOp_1:value:0*
_output_shapes
: *
Toutput_types
2:????
OptionalFromValue_4OptionalFromValue"FusedBatchNormV3:reserve_space_1:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_5OptionalFromValue"FusedBatchNormV3:reserve_space_2:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_6OptionalFromValue"FusedBatchNormV3:reserve_space_3:0*
_output_shapes
: *
Toutput_types
2:???"3
fusedbatchnormv3_0FusedBatchNormV3:batch_mean:0"7
fusedbatchnormv3_1!FusedBatchNormV3:batch_variance:0"1
optionalfromvalueOptionalFromValue:optional:0"(
fusedbatchnormv3FusedBatchNormV3:y:0"5
optionalfromvalue_1OptionalFromValue_1:optional:0"5
optionalfromvalue_2OptionalFromValue_2:optional:0"5
optionalfromvalue_3OptionalFromValue_3:optional:0"5
optionalfromvalue_4OptionalFromValue_4:optional:0"5
optionalfromvalue_5OptionalFromValue_5:optional:0"5
optionalfromvalue_6OptionalFromValue_6:optional:0*-
_input_shapes
: : : : :{:,(
&
_output_shapes
:{
?
?
$batch_normalization_2_cond_false_2908
*readvariableop_batch_normalization_2_gamma:9
+readvariableop_1_batch_normalization_2_beta:O
Afusedbatchnormv3_readvariableop_batch_normalization_2_moving_mean:U
Gfusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance:8
4fusedbatchnormv3_stream_2_depthwise_conv2d_1_biasadd
fusedbatchnormv3
fusedbatchnormv3_0
fusedbatchnormv3_1
optionalfromvalue
optionalfromvalue_1
optionalfromvalue_2
optionalfromvalue_3
optionalfromvalue_4
optionalfromvalue_5
optionalfromvalue_6?u
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_2_gamma*
dtype0*
_output_shapes
:x
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_2_beta*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_2_moving_mean*
dtype0*
_output_shapes
:?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance*
dtype0*
_output_shapes
:?
FusedBatchNormV3FusedBatchNormV34fusedbatchnormv3_stream_2_depthwise_conv2d_1_biasaddReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
exponential_avg_factor%  ??*
U0*
epsilon%??'7*B
_output_shapes0
.:9:::::*
data_formatNHWC*
T0*
is_training( ?
OptionalFromValueOptionalFromValueReadVariableOp:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_1OptionalFromValueReadVariableOp_1:value:0*
_output_shapes
: *
Toutput_types
2:????
OptionalFromValue_2OptionalFromValue'FusedBatchNormV3/ReadVariableOp:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_3OptionalFromValue)FusedBatchNormV3/ReadVariableOp_1:value:0*
_output_shapes
: *
Toutput_types
2:????
OptionalFromValue_4OptionalFromValue"FusedBatchNormV3:reserve_space_1:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_5OptionalFromValue"FusedBatchNormV3:reserve_space_2:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_6OptionalFromValue"FusedBatchNormV3:reserve_space_3:0*
Toutput_types
2*
_output_shapes
: :???"3
fusedbatchnormv3_0FusedBatchNormV3:batch_mean:0"7
fusedbatchnormv3_1!FusedBatchNormV3:batch_variance:0"1
optionalfromvalueOptionalFromValue:optional:0"(
fusedbatchnormv3FusedBatchNormV3:y:0"5
optionalfromvalue_1OptionalFromValue_1:optional:0"5
optionalfromvalue_2OptionalFromValue_2:optional:0"5
optionalfromvalue_3OptionalFromValue_3:optional:0"5
optionalfromvalue_4OptionalFromValue_4:optional:0"5
optionalfromvalue_5OptionalFromValue_5:optional:0"5
optionalfromvalue_6OptionalFromValue_6:optional:0*-
_input_shapes
: : : : :9:,(
&
_output_shapes
:9
?
?
#batch_normalization_1_cond_true_1648
*readvariableop_batch_normalization_1_gamma:9
+readvariableop_1_batch_normalization_1_beta:O
Afusedbatchnormv3_readvariableop_batch_normalization_1_moving_mean:U
Gfusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance:6
2fusedbatchnormv3_stream_1_depthwise_conv2d_biasadd
fusedbatchnormv3
fusedbatchnormv3_0
fusedbatchnormv3_1
optionalfromvalue
optionalfromvalue_1
optionalfromvalue_2
optionalfromvalue_3
optionalfromvalue_4
optionalfromvalue_5
optionalfromvalue_6?u
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_1_gamma*
dtype0*
_output_shapes
:x
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_1_beta*
dtype0*
_output_shapes
:?
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_1_moving_mean*
dtype0*
_output_shapes
:?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance*
dtype0*
_output_shapes
:?
FusedBatchNormV3FusedBatchNormV32fusedbatchnormv3_stream_1_depthwise_conv2d_biasaddReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
U0*
exponential_avg_factor%  ??*
data_formatNHWC*
is_training(*
T0*
epsilon%??'7*B
_output_shapes0
.:<:::::?
OptionalFromValueOptionalFromValueReadVariableOp:value:0*
_output_shapes
: *
Toutput_types
2:????
OptionalFromValue_1OptionalFromValueReadVariableOp_1:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_2OptionalFromValue'FusedBatchNormV3/ReadVariableOp:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_3OptionalFromValue)FusedBatchNormV3/ReadVariableOp_1:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_4OptionalFromValue"FusedBatchNormV3:reserve_space_1:0*
_output_shapes
: *
Toutput_types
2:????
OptionalFromValue_5OptionalFromValue"FusedBatchNormV3:reserve_space_2:0*
_output_shapes
: *
Toutput_types
2:????
OptionalFromValue_6OptionalFromValue"FusedBatchNormV3:reserve_space_3:0*
_output_shapes
: *
Toutput_types
2:???"3
fusedbatchnormv3_0FusedBatchNormV3:batch_mean:0"7
fusedbatchnormv3_1!FusedBatchNormV3:batch_variance:0"1
optionalfromvalueOptionalFromValue:optional:0"(
fusedbatchnormv3FusedBatchNormV3:y:0"5
optionalfromvalue_1OptionalFromValue_1:optional:0"5
optionalfromvalue_2OptionalFromValue_2:optional:0"5
optionalfromvalue_3OptionalFromValue_3:optional:0"5
optionalfromvalue_4OptionalFromValue_4:optional:0"5
optionalfromvalue_5OptionalFromValue_5:optional:0"5
optionalfromvalue_6OptionalFromValue_6:optional:0*-
_input_shapes
: : : : :<:,(
&
_output_shapes
:<
?
?
dropout_cond_true_629!
dropout_mul_activation_4_relu
dropout_mul_1
optionalfromvalue
optionalfromvalue_1
optionalfromvalue_2
optionalfromvalue_3
optionalfromvalue_4
optionalfromvalue_5
optionalfromvalue_6?R
dropout/ConstConst*
valueB
 *   @*
_output_shapes
: *
dtype0z
dropout/MulMuldropout_mul_activation_4_reludropout/Const:output:0*
T0*&
_output_shapes
:f
dropout/ShapeConst*%
valueB"            *
_output_shapes
:*
dtype0?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
seed2 *

seed *
T0*
dtype0*&
_output_shapes
:[
dropout/GreaterEqual/yConst*
dtype0*
_output_shapes
: *
valueB
 *   ??
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*&
_output_shapes
:~
dropout/CastCastdropout/GreaterEqual:z:0*
Truncate( *&
_output_shapes
:*

DstT0*

SrcT0
h
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*&
_output_shapes
:?
OptionalFromValueOptionalFromValuedropout/Const:output:0*
_output_shapes
: *
Toutput_types
2:????
OptionalFromValue_1OptionalFromValuedropout/Mul:z:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_2OptionalFromValuedropout/Shape:output:0*
_output_shapes
: *
Toutput_types
2:????
OptionalFromValue_3OptionalFromValue-dropout/random_uniform/RandomUniform:output:0*
_output_shapes
: *
Toutput_types
2:????
OptionalFromValue_4OptionalFromValuedropout/GreaterEqual/y:output:0*
_output_shapes
: *
Toutput_types
2:????
OptionalFromValue_5OptionalFromValuedropout/GreaterEqual:z:0*
_output_shapes
: *
Toutput_types
2
:????
OptionalFromValue_6OptionalFromValuedropout/Cast:y:0*
Toutput_types
2*
_output_shapes
: :???"5
optionalfromvalue_2OptionalFromValue_2:optional:0"5
optionalfromvalue_3OptionalFromValue_3:optional:0"5
optionalfromvalue_4OptionalFromValue_4:optional:0"5
optionalfromvalue_5OptionalFromValue_5:optional:0"5
optionalfromvalue_6OptionalFromValue_6:optional:0""
dropout_mul_1dropout/Mul_1:z:0"1
optionalfromvalueOptionalFromValue:optional:0"5
optionalfromvalue_1OptionalFromValue_1:optional:0*%
_input_shapes
::, (
&
_output_shapes
:
?
3
&batch_normalization_2_cond_1_false_358	
constJ
ConstConst*
dtype0*
_output_shapes
: *
valueB
 *  ??"
constConst:output:0*
_input_shapes 
?
?
!batch_normalization_cond_false_406
(readvariableop_batch_normalization_gamma:7
)readvariableop_1_batch_normalization_beta:M
?fusedbatchnormv3_readvariableop_batch_normalization_moving_mean:S
Efusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance:*
&fusedbatchnormv3_stream_conv2d_biasadd
fusedbatchnormv3
fusedbatchnormv3_0
fusedbatchnormv3_1
optionalfromvalue
optionalfromvalue_1
optionalfromvalue_2
optionalfromvalue_3
optionalfromvalue_4
optionalfromvalue_5
optionalfromvalue_6?s
ReadVariableOpReadVariableOp(readvariableop_batch_normalization_gamma*
_output_shapes
:*
dtype0v
ReadVariableOp_1ReadVariableOp)readvariableop_1_batch_normalization_beta*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp?fusedbatchnormv3_readvariableop_batch_normalization_moving_mean*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpEfusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3&fusedbatchnormv3_stream_conv2d_biasaddReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
epsilon%??'7*
T0*
U0*
is_training( *B
_output_shapes0
.:{:::::*
data_formatNHWC*
exponential_avg_factor%  ???
OptionalFromValueOptionalFromValueReadVariableOp:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_1OptionalFromValueReadVariableOp_1:value:0*
_output_shapes
: *
Toutput_types
2:????
OptionalFromValue_2OptionalFromValue'FusedBatchNormV3/ReadVariableOp:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_3OptionalFromValue)FusedBatchNormV3/ReadVariableOp_1:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_4OptionalFromValue"FusedBatchNormV3:reserve_space_1:0*
_output_shapes
: *
Toutput_types
2:????
OptionalFromValue_5OptionalFromValue"FusedBatchNormV3:reserve_space_2:0*
_output_shapes
: *
Toutput_types
2:????
OptionalFromValue_6OptionalFromValue"FusedBatchNormV3:reserve_space_3:0*
_output_shapes
: *
Toutput_types
2:???"(
fusedbatchnormv3FusedBatchNormV3:y:0"5
optionalfromvalue_1OptionalFromValue_1:optional:0"5
optionalfromvalue_2OptionalFromValue_2:optional:0"5
optionalfromvalue_3OptionalFromValue_3:optional:0"5
optionalfromvalue_4OptionalFromValue_4:optional:0"5
optionalfromvalue_5OptionalFromValue_5:optional:0"5
optionalfromvalue_6OptionalFromValue_6:optional:0"3
fusedbatchnormv3_0FusedBatchNormV3:batch_mean:0"7
fusedbatchnormv3_1!FusedBatchNormV3:batch_variance:0"1
optionalfromvalueOptionalFromValue:optional:0*-
_input_shapes
: : : : :{:,(
&
_output_shapes
:{
?
?
#batch_normalization_3_cond_true_4158
*readvariableop_batch_normalization_3_gamma:9
+readvariableop_1_batch_normalization_3_beta:O
Afusedbatchnormv3_readvariableop_batch_normalization_3_moving_mean:U
Gfusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance:8
4fusedbatchnormv3_stream_4_depthwise_conv2d_2_biasadd
fusedbatchnormv3
fusedbatchnormv3_0
fusedbatchnormv3_1
optionalfromvalue
optionalfromvalue_1
optionalfromvalue_2
optionalfromvalue_3
optionalfromvalue_4
optionalfromvalue_5
optionalfromvalue_6?u
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_3_gamma*
dtype0*
_output_shapes
:x
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_3_beta*
dtype0*
_output_shapes
:?
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_3_moving_mean*
dtype0*
_output_shapes
:?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance*
dtype0*
_output_shapes
:?
FusedBatchNormV3FusedBatchNormV34fusedbatchnormv3_stream_4_depthwise_conv2d_2_biasaddReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*
epsilon%??'7*
is_training(*B
_output_shapes0
.::::::*
exponential_avg_factor%  ??*
data_formatNHWC?
OptionalFromValueOptionalFromValueReadVariableOp:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_1OptionalFromValueReadVariableOp_1:value:0*
_output_shapes
: *
Toutput_types
2:????
OptionalFromValue_2OptionalFromValue'FusedBatchNormV3/ReadVariableOp:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_3OptionalFromValue)FusedBatchNormV3/ReadVariableOp_1:value:0*
_output_shapes
: *
Toutput_types
2:????
OptionalFromValue_4OptionalFromValue"FusedBatchNormV3:reserve_space_1:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_5OptionalFromValue"FusedBatchNormV3:reserve_space_2:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_6OptionalFromValue"FusedBatchNormV3:reserve_space_3:0*
Toutput_types
2*
_output_shapes
: :???"7
fusedbatchnormv3_1!FusedBatchNormV3:batch_variance:0"1
optionalfromvalueOptionalFromValue:optional:0"(
fusedbatchnormv3FusedBatchNormV3:y:0"5
optionalfromvalue_1OptionalFromValue_1:optional:0"5
optionalfromvalue_2OptionalFromValue_2:optional:0"5
optionalfromvalue_3OptionalFromValue_3:optional:0"5
optionalfromvalue_4OptionalFromValue_4:optional:0"5
optionalfromvalue_5OptionalFromValue_5:optional:0"5
optionalfromvalue_6OptionalFromValue_6:optional:0"3
fusedbatchnormv3_0FusedBatchNormV3:batch_mean:0*-
_input_shapes
: : : : ::,(
&
_output_shapes
:
?
0
#batch_normalization_cond_1_true_107	
constJ
ConstConst*
_output_shapes
: *
valueB
 *???=*
dtype0"
constConst:output:0*
_input_shapes 
?
1
$batch_normalization_cond_1_false_108	
constJ
ConstConst*
_output_shapes
: *
valueB
 *  ??*
dtype0"
constConst:output:0*
_input_shapes 
?
2
%batch_normalization_1_cond_1_true_232	
constJ
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *???="
constConst:output:0*
_input_shapes 
?
?
$batch_normalization_4_cond_false_5418
*readvariableop_batch_normalization_4_gamma:9
+readvariableop_1_batch_normalization_4_beta:O
Afusedbatchnormv3_readvariableop_batch_normalization_4_moving_mean:U
Gfusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance:8
4fusedbatchnormv3_stream_5_depthwise_conv2d_3_biasadd
fusedbatchnormv3
fusedbatchnormv3_0
fusedbatchnormv3_1
optionalfromvalue
optionalfromvalue_1
optionalfromvalue_2
optionalfromvalue_3
optionalfromvalue_4
optionalfromvalue_5
optionalfromvalue_6?u
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_4_gamma*
_output_shapes
:*
dtype0x
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_4_beta*
dtype0*
_output_shapes
:?
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_4_moving_mean*
dtype0*
_output_shapes
:?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance*
dtype0*
_output_shapes
:?
FusedBatchNormV3FusedBatchNormV34fusedbatchnormv3_stream_5_depthwise_conv2d_3_biasaddReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
epsilon%??'7*
exponential_avg_factor%  ??*
data_formatNHWC*
is_training( *
T0*B
_output_shapes0
.::::::*
U0?
OptionalFromValueOptionalFromValueReadVariableOp:value:0*
_output_shapes
: *
Toutput_types
2:????
OptionalFromValue_1OptionalFromValueReadVariableOp_1:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_2OptionalFromValue'FusedBatchNormV3/ReadVariableOp:value:0*
_output_shapes
: *
Toutput_types
2:????
OptionalFromValue_3OptionalFromValue)FusedBatchNormV3/ReadVariableOp_1:value:0*
_output_shapes
: *
Toutput_types
2:????
OptionalFromValue_4OptionalFromValue"FusedBatchNormV3:reserve_space_1:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_5OptionalFromValue"FusedBatchNormV3:reserve_space_2:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_6OptionalFromValue"FusedBatchNormV3:reserve_space_3:0*
_output_shapes
: *
Toutput_types
2:???"3
fusedbatchnormv3_0FusedBatchNormV3:batch_mean:0"7
fusedbatchnormv3_1!FusedBatchNormV3:batch_variance:0"1
optionalfromvalueOptionalFromValue:optional:0"(
fusedbatchnormv3FusedBatchNormV3:y:0"5
optionalfromvalue_1OptionalFromValue_1:optional:0"5
optionalfromvalue_2OptionalFromValue_2:optional:0"5
optionalfromvalue_3OptionalFromValue_3:optional:0"5
optionalfromvalue_4OptionalFromValue_4:optional:0"5
optionalfromvalue_5OptionalFromValue_5:optional:0"5
optionalfromvalue_6OptionalFromValue_6:optional:0*-
_input_shapes
: : : : ::,(
&
_output_shapes
:
?
?
dropout_cond_false_630
identity_activation_4_relu
identity
optionalnone
optionalnone_1
optionalnone_2
optionalnone_3
optionalnone_4
optionalnone_5
optionalnone_6a
IdentityIdentityidentity_activation_4_relu*&
_output_shapes
:*
T04
OptionalNoneOptionalNone*
_output_shapes
: 6
OptionalNone_1OptionalNone*
_output_shapes
: 6
OptionalNone_2OptionalNone*
_output_shapes
: 6
OptionalNone_3OptionalNone*
_output_shapes
: 6
OptionalNone_4OptionalNone*
_output_shapes
: 6
OptionalNone_5OptionalNone*
_output_shapes
: 6
OptionalNone_6OptionalNone*
_output_shapes
: "+
optionalnone_5OptionalNone_5:optional:0"'
optionalnoneOptionalNone:optional:0"+
optionalnone_6OptionalNone_6:optional:0"
identityIdentity:output:0"+
optionalnone_1OptionalNone_1:optional:0"+
optionalnone_2OptionalNone_2:optional:0"+
optionalnone_3OptionalNone_3:optional:0"+
optionalnone_4OptionalNone_4:optional:0*%
_input_shapes
::, (
&
_output_shapes
:
?
?
$batch_normalization_1_cond_false_1658
*readvariableop_batch_normalization_1_gamma:9
+readvariableop_1_batch_normalization_1_beta:O
Afusedbatchnormv3_readvariableop_batch_normalization_1_moving_mean:U
Gfusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance:6
2fusedbatchnormv3_stream_1_depthwise_conv2d_biasadd
fusedbatchnormv3
fusedbatchnormv3_0
fusedbatchnormv3_1
optionalfromvalue
optionalfromvalue_1
optionalfromvalue_2
optionalfromvalue_3
optionalfromvalue_4
optionalfromvalue_5
optionalfromvalue_6?u
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_1_gamma*
_output_shapes
:*
dtype0x
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_1_beta*
dtype0*
_output_shapes
:?
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_1_moving_mean*
dtype0*
_output_shapes
:?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV32fusedbatchnormv3_stream_1_depthwise_conv2d_biasaddReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
is_training( *
epsilon%??'7*
U0*
data_formatNHWC*
exponential_avg_factor%  ??*B
_output_shapes0
.:<:::::*
T0?
OptionalFromValueOptionalFromValueReadVariableOp:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_1OptionalFromValueReadVariableOp_1:value:0*
_output_shapes
: *
Toutput_types
2:????
OptionalFromValue_2OptionalFromValue'FusedBatchNormV3/ReadVariableOp:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_3OptionalFromValue)FusedBatchNormV3/ReadVariableOp_1:value:0*
_output_shapes
: *
Toutput_types
2:????
OptionalFromValue_4OptionalFromValue"FusedBatchNormV3:reserve_space_1:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_5OptionalFromValue"FusedBatchNormV3:reserve_space_2:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_6OptionalFromValue"FusedBatchNormV3:reserve_space_3:0*
Toutput_types
2*
_output_shapes
: :???"5
optionalfromvalue_2OptionalFromValue_2:optional:0"5
optionalfromvalue_3OptionalFromValue_3:optional:0"5
optionalfromvalue_4OptionalFromValue_4:optional:0"5
optionalfromvalue_5OptionalFromValue_5:optional:0"5
optionalfromvalue_6OptionalFromValue_6:optional:0"3
fusedbatchnormv3_0FusedBatchNormV3:batch_mean:0"7
fusedbatchnormv3_1!FusedBatchNormV3:batch_variance:0"1
optionalfromvalueOptionalFromValue:optional:0"(
fusedbatchnormv3FusedBatchNormV3:y:0"5
optionalfromvalue_1OptionalFromValue_1:optional:0*-
_input_shapes
: : : : :<:,(
&
_output_shapes
:<
?
2
%batch_normalization_3_cond_1_true_483	
constJ
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *???="
constConst:output:0*
_input_shapes 
?
3
&batch_normalization_4_cond_1_false_609	
constJ
ConstConst*
valueB
 *  ??*
_output_shapes
: *
dtype0"
constConst:output:0*
_input_shapes "??&Jf?     ȾG	? ?N???AJ??
??
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
E
AssignSubVariableOp
resource
value"dtype"
dtypetype?
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ?
?
AvgPool

value"T
output"T"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW"
Ttype:
2
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

?
DepthwiseConv2dNative

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
?
If
cond"Tcond
input2Tin
output2Tout"
Tcondtype"
Tin
list(type)("
Tout
list(type)("
then_branchfunc"
else_branchfunc" 
output_shapeslist(shape)
 ?
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
?
Mul
x"T
y"T
z"T"
Ttype:
2	?
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	?
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
9
Softmax
logits"T
softmax"T"
Ttype:
2
?
StatelessIf
cond"Tcond
input2Tin
output2Tout"
Tcondtype"
Tin
list(type)("
Tout
list(type)("
then_branchfunc"
else_branchfunc" 
output_shapeslist(shape)
 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?
9
VarIsInitializedOp
resource
is_initialized
?*2.9.02v2.9.0-rc2-42-g8a20d54a3c1??
j
input_1Placeholder*'
_output_shapes
:?	*
shape:?	*
dtype0
?
max_pooling2d/MaxPoolMaxPoolinput_1*
data_formatNHWC*
T0*
explicit_paddings
 *
paddingVALID*'
_output_shapes
:?*
ksize
*
strides

?
.stream/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*
dtype0* 
_class
loc:@stream/kernel*%
valueB"            
?
,stream/kernel/Initializer/random_uniform/minConst*
dtype0*
valueB
 *   ?* 
_class
loc:@stream/kernel*
_output_shapes
: 
?
,stream/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *
dtype0* 
_class
loc:@stream/kernel*
valueB
 *   ?
?
6stream/kernel/Initializer/random_uniform/RandomUniformRandomUniform.stream/kernel/Initializer/random_uniform/shape*
dtype0*

seed *
seed2 * 
_class
loc:@stream/kernel*&
_output_shapes
:*
T0
?
,stream/kernel/Initializer/random_uniform/subSub,stream/kernel/Initializer/random_uniform/max,stream/kernel/Initializer/random_uniform/min* 
_class
loc:@stream/kernel*
_output_shapes
: *
T0
?
,stream/kernel/Initializer/random_uniform/mulMul6stream/kernel/Initializer/random_uniform/RandomUniform,stream/kernel/Initializer/random_uniform/sub*&
_output_shapes
:* 
_class
loc:@stream/kernel*
T0
?
(stream/kernel/Initializer/random_uniformAddV2,stream/kernel/Initializer/random_uniform/mul,stream/kernel/Initializer/random_uniform/min*&
_output_shapes
:*
T0* 
_class
loc:@stream/kernel
?
stream/kernelVarHandleOp* 
_class
loc:@stream/kernel*
dtype0*
shared_namestream/kernel*
allowed_devices
 *
_output_shapes
: *
shape:*
	container 
k
.stream/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpstream/kernel*
_output_shapes
: 
?
stream/kernel/AssignAssignVariableOpstream/kernel(stream/kernel/Initializer/random_uniform*
dtype0*
validate_shape( 
w
!stream/kernel/Read/ReadVariableOpReadVariableOpstream/kernel*
dtype0*&
_output_shapes
:
?
stream/bias/Initializer/zerosConst*
dtype0*
_output_shapes
:*
valueB*    *
_class
loc:@stream/bias
?
stream/biasVarHandleOp*
dtype0*
shared_namestream/bias*
allowed_devices
 *
_class
loc:@stream/bias*
shape:*
	container *
_output_shapes
: 
g
,stream/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpstream/bias*
_output_shapes
: 
u
stream/bias/AssignAssignVariableOpstream/biasstream/bias/Initializer/zeros*
dtype0*
validate_shape( 
g
stream/bias/Read/ReadVariableOpReadVariableOpstream/bias*
_output_shapes
:*
dtype0
y
#stream/conv2d/Conv2D/ReadVariableOpReadVariableOpstream/kernel*&
_output_shapes
:*
dtype0
?
stream/conv2d/Conv2DConv2Dmax_pooling2d/MaxPool#stream/conv2d/Conv2D/ReadVariableOp*&
_output_shapes
:{*
strides
*
	dilations
*
paddingVALID*
explicit_paddings
 *
data_formatNHWC*
T0*
use_cudnn_on_gpu(
l
$stream/conv2d/BiasAdd/ReadVariableOpReadVariableOpstream/bias*
_output_shapes
:*
dtype0
?
stream/conv2d/BiasAddBiasAddstream/conv2d/Conv2D$stream/conv2d/BiasAdd/ReadVariableOp*&
_output_shapes
:{*
data_formatNHWC*
T0
?
*batch_normalization/gamma/Initializer/onesConst*
_output_shapes
:*
dtype0*,
_class"
 loc:@batch_normalization/gamma*
valueB*  ??
?
batch_normalization/gammaVarHandleOp*,
_class"
 loc:@batch_normalization/gamma*
	container *
dtype0*
_output_shapes
: *
allowed_devices
 **
shared_namebatch_normalization/gamma*
shape:
?
:batch_normalization/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization/gamma*
_output_shapes
: 
?
 batch_normalization/gamma/AssignAssignVariableOpbatch_normalization/gamma*batch_normalization/gamma/Initializer/ones*
dtype0*
validate_shape( 
?
-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes
:*
dtype0
?
*batch_normalization/beta/Initializer/zerosConst*
dtype0*
_output_shapes
:*+
_class!
loc:@batch_normalization/beta*
valueB*    
?
batch_normalization/betaVarHandleOp*
shape:*
_output_shapes
: *
dtype0*+
_class!
loc:@batch_normalization/beta*
allowed_devices
 *
	container *)
shared_namebatch_normalization/beta
?
9batch_normalization/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization/beta*
_output_shapes
: 
?
batch_normalization/beta/AssignAssignVariableOpbatch_normalization/beta*batch_normalization/beta/Initializer/zeros*
dtype0*
validate_shape( 
?
,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes
:*
dtype0
?
1batch_normalization/moving_mean/Initializer/zerosConst*
valueB*    *
_output_shapes
:*
dtype0*2
_class(
&$loc:@batch_normalization/moving_mean
?
batch_normalization/moving_meanVarHandleOp*
	container *2
_class(
&$loc:@batch_normalization/moving_mean*
allowed_devices
 *0
shared_name!batch_normalization/moving_mean*
shape:*
_output_shapes
: *
dtype0
?
@batch_normalization/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization/moving_mean*
_output_shapes
: 
?
&batch_normalization/moving_mean/AssignAssignVariableOpbatch_normalization/moving_mean1batch_normalization/moving_mean/Initializer/zeros*
dtype0*
validate_shape( 
?
3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes
:*
dtype0
?
4batch_normalization/moving_variance/Initializer/onesConst*6
_class,
*(loc:@batch_normalization/moving_variance*
_output_shapes
:*
dtype0*
valueB*  ??
?
#batch_normalization/moving_varianceVarHandleOp*
shape:*4
shared_name%#batch_normalization/moving_variance*
allowed_devices
 *6
_class,
*(loc:@batch_normalization/moving_variance*
dtype0*
	container *
_output_shapes
: 
?
Dbatch_normalization/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp#batch_normalization/moving_variance*
_output_shapes
: 
?
*batch_normalization/moving_variance/AssignAssignVariableOp#batch_normalization/moving_variance4batch_normalization/moving_variance/Initializer/ones*
dtype0*
validate_shape( 
?
7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes
:*
dtype0
\
keras_learning_phase/inputConst*
value	B
 Z *
_output_shapes
: *
dtype0

|
keras_learning_phasePlaceholderWithDefaultkeras_learning_phase/input*
dtype0
*
shape: *
_output_shapes
: 
?
batch_normalization/condIfkeras_learning_phasebatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_variancestream/conv2d/BiasAdd*3
then_branch$R"
 batch_normalization_cond_true_39*&
_read_only_resource_inputs
*
_lower_using_switch_merge(*
Tcond0
*
Tin	
2*4
else_branch%R#
!batch_normalization_cond_false_40*@
_output_shapes.
,:{::: : : : : : : *
Tout
2
*?
output_shapes.
,:{::: : : : : : : 
x
!batch_normalization/cond/IdentityIdentitybatch_normalization/cond*&
_output_shapes
:{*
T0
p
#batch_normalization/cond/Identity_1Identitybatch_normalization/cond:1*
_output_shapes
:*
T0
p
#batch_normalization/cond/Identity_2Identitybatch_normalization/cond:2*
_output_shapes
:*
T0
l
#batch_normalization/cond/Identity_3Identitybatch_normalization/cond:3*
_output_shapes
: *
T0
l
#batch_normalization/cond/Identity_4Identitybatch_normalization/cond:4*
T0*
_output_shapes
: 
l
#batch_normalization/cond/Identity_5Identitybatch_normalization/cond:5*
T0*
_output_shapes
: 
l
#batch_normalization/cond/Identity_6Identitybatch_normalization/cond:6*
_output_shapes
: *
T0
l
#batch_normalization/cond/Identity_7Identitybatch_normalization/cond:7*
_output_shapes
: *
T0
l
#batch_normalization/cond/Identity_8Identitybatch_normalization/cond:8*
T0*
_output_shapes
: 
l
#batch_normalization/cond/Identity_9Identitybatch_normalization/cond:9*
_output_shapes
: *
T0
?
batch_normalization/cond_1StatelessIfkeras_learning_phase*6
then_branch'R%
#batch_normalization_cond_1_true_107*7
else_branch(R&
$batch_normalization_cond_1_false_108*
Tcond0
*
_lower_using_switch_merge(*
_output_shapes
: *
Tout
2*	
Tin
 * 
_read_only_resource_inputs
 *
output_shapes
: 
l
#batch_normalization/cond_1/IdentityIdentitybatch_normalization/cond_1*
T0*
_output_shapes
: 
?
)batch_normalization/AssignMovingAvg/sub/xConst*
_output_shapes
: *
valueB
 *  ??*
dtype0*2
_class(
&$loc:@batch_normalization/moving_mean
?
'batch_normalization/AssignMovingAvg/subSub)batch_normalization/AssignMovingAvg/sub/x#batch_normalization/cond_1/Identity*
_output_shapes
: *2
_class(
&$loc:@batch_normalization/moving_mean*
T0
?
2batch_normalization/AssignMovingAvg/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
dtype0*
_output_shapes
:
?
)batch_normalization/AssignMovingAvg/sub_1Sub2batch_normalization/AssignMovingAvg/ReadVariableOp#batch_normalization/cond/Identity_1*
_output_shapes
:*
T0*2
_class(
&$loc:@batch_normalization/moving_mean
?
'batch_normalization/AssignMovingAvg/mulMul)batch_normalization/AssignMovingAvg/sub_1'batch_normalization/AssignMovingAvg/sub*
T0*
_output_shapes
:*2
_class(
&$loc:@batch_normalization/moving_mean
?
7batch_normalization/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpbatch_normalization/moving_mean'batch_normalization/AssignMovingAvg/mul*
dtype0*2
_class(
&$loc:@batch_normalization/moving_mean
?
4batch_normalization/AssignMovingAvg/ReadVariableOp_1ReadVariableOpbatch_normalization/moving_mean8^batch_normalization/AssignMovingAvg/AssignSubVariableOp*2
_class(
&$loc:@batch_normalization/moving_mean*
_output_shapes
:*
dtype0
?
+batch_normalization/AssignMovingAvg_1/sub/xConst*6
_class,
*(loc:@batch_normalization/moving_variance*
_output_shapes
: *
dtype0*
valueB
 *  ??
?
)batch_normalization/AssignMovingAvg_1/subSub+batch_normalization/AssignMovingAvg_1/sub/x#batch_normalization/cond_1/Identity*
T0*
_output_shapes
: *6
_class,
*(loc:@batch_normalization/moving_variance
?
4batch_normalization/AssignMovingAvg_1/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes
:*
dtype0
?
+batch_normalization/AssignMovingAvg_1/sub_1Sub4batch_normalization/AssignMovingAvg_1/ReadVariableOp#batch_normalization/cond/Identity_2*
T0*
_output_shapes
:*6
_class,
*(loc:@batch_normalization/moving_variance
?
)batch_normalization/AssignMovingAvg_1/mulMul+batch_normalization/AssignMovingAvg_1/sub_1)batch_normalization/AssignMovingAvg_1/sub*6
_class,
*(loc:@batch_normalization/moving_variance*
T0*
_output_shapes
:
?
9batch_normalization/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp#batch_normalization/moving_variance)batch_normalization/AssignMovingAvg_1/mul*6
_class,
*(loc:@batch_normalization/moving_variance*
dtype0
?
6batch_normalization/AssignMovingAvg_1/ReadVariableOp_1ReadVariableOp#batch_normalization/moving_variance:^batch_normalization/AssignMovingAvg_1/AssignSubVariableOp*
dtype0*
_output_shapes
:*6
_class,
*(loc:@batch_normalization/moving_variance
k
activation/ReluRelu!batch_normalization/cond/Identity*
T0*&
_output_shapes
:{
?
:stream_1/depthwise_kernel/Initializer/random_uniform/shapeConst*,
_class"
 loc:@stream_1/depthwise_kernel*
dtype0*
_output_shapes
:*%
valueB"            
?
8stream_1/depthwise_kernel/Initializer/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *?7?*,
_class"
 loc:@stream_1/depthwise_kernel
?
8stream_1/depthwise_kernel/Initializer/random_uniform/maxConst*
dtype0*
valueB
 *?7?*
_output_shapes
: *,
_class"
 loc:@stream_1/depthwise_kernel
?
Bstream_1/depthwise_kernel/Initializer/random_uniform/RandomUniformRandomUniform:stream_1/depthwise_kernel/Initializer/random_uniform/shape*,
_class"
 loc:@stream_1/depthwise_kernel*&
_output_shapes
:*

seed *
dtype0*
T0*
seed2 
?
8stream_1/depthwise_kernel/Initializer/random_uniform/subSub8stream_1/depthwise_kernel/Initializer/random_uniform/max8stream_1/depthwise_kernel/Initializer/random_uniform/min*
T0*,
_class"
 loc:@stream_1/depthwise_kernel*
_output_shapes
: 
?
8stream_1/depthwise_kernel/Initializer/random_uniform/mulMulBstream_1/depthwise_kernel/Initializer/random_uniform/RandomUniform8stream_1/depthwise_kernel/Initializer/random_uniform/sub*,
_class"
 loc:@stream_1/depthwise_kernel*
T0*&
_output_shapes
:
?
4stream_1/depthwise_kernel/Initializer/random_uniformAddV28stream_1/depthwise_kernel/Initializer/random_uniform/mul8stream_1/depthwise_kernel/Initializer/random_uniform/min*
T0*,
_class"
 loc:@stream_1/depthwise_kernel*&
_output_shapes
:
?
stream_1/depthwise_kernelVarHandleOp*
dtype0*
allowed_devices
 **
shared_namestream_1/depthwise_kernel*,
_class"
 loc:@stream_1/depthwise_kernel*
shape:*
	container *
_output_shapes
: 
?
:stream_1/depthwise_kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpstream_1/depthwise_kernel*
_output_shapes
: 
?
 stream_1/depthwise_kernel/AssignAssignVariableOpstream_1/depthwise_kernel4stream_1/depthwise_kernel/Initializer/random_uniform*
dtype0*
validate_shape( 
?
-stream_1/depthwise_kernel/Read/ReadVariableOpReadVariableOpstream_1/depthwise_kernel*
dtype0*&
_output_shapes
:
?
stream_1/bias/Initializer/zerosConst*
_output_shapes
:*
valueB*    *
dtype0* 
_class
loc:@stream_1/bias
?
stream_1/biasVarHandleOp*
shared_namestream_1/bias*
shape:*
allowed_devices
 * 
_class
loc:@stream_1/bias*
dtype0*
_output_shapes
: *
	container 
k
.stream_1/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpstream_1/bias*
_output_shapes
: 
{
stream_1/bias/AssignAssignVariableOpstream_1/biasstream_1/bias/Initializer/zeros*
dtype0*
validate_shape( 
k
!stream_1/bias/Read/ReadVariableOpReadVariableOpstream_1/bias*
dtype0*
_output_shapes
:
?
2stream_1/depthwise_conv2d/depthwise/ReadVariableOpReadVariableOpstream_1/depthwise_kernel*
dtype0*&
_output_shapes
:
?
)stream_1/depthwise_conv2d/depthwise/ShapeConst*
dtype0*%
valueB"            *
_output_shapes
:
?
1stream_1/depthwise_conv2d/depthwise/dilation_rateConst*
dtype0*
valueB"      *
_output_shapes
:
?
#stream_1/depthwise_conv2d/depthwiseDepthwiseConv2dNativeactivation/Relu2stream_1/depthwise_conv2d/depthwise/ReadVariableOp*
T0*&
_output_shapes
:<*
strides
*
	dilations
*
explicit_paddings
 *
data_formatNHWC*
paddingVALID
z
0stream_1/depthwise_conv2d/BiasAdd/ReadVariableOpReadVariableOpstream_1/bias*
_output_shapes
:*
dtype0
?
!stream_1/depthwise_conv2d/BiasAddBiasAdd#stream_1/depthwise_conv2d/depthwise0stream_1/depthwise_conv2d/BiasAdd/ReadVariableOp*&
_output_shapes
:<*
T0*
data_formatNHWC
?
,batch_normalization_1/gamma/Initializer/onesConst*
valueB*  ??*
_output_shapes
:*
dtype0*.
_class$
" loc:@batch_normalization_1/gamma
?
batch_normalization_1/gammaVarHandleOp*
dtype0*
_output_shapes
: *
	container *
shape:*,
shared_namebatch_normalization_1/gamma*.
_class$
" loc:@batch_normalization_1/gamma*
allowed_devices
 
?
<batch_normalization_1/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_1/gamma*
_output_shapes
: 
?
"batch_normalization_1/gamma/AssignAssignVariableOpbatch_normalization_1/gamma,batch_normalization_1/gamma/Initializer/ones*
dtype0*
validate_shape( 
?
/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
dtype0*
_output_shapes
:
?
,batch_normalization_1/beta/Initializer/zerosConst*-
_class#
!loc:@batch_normalization_1/beta*
_output_shapes
:*
valueB*    *
dtype0
?
batch_normalization_1/betaVarHandleOp*+
shared_namebatch_normalization_1/beta*-
_class#
!loc:@batch_normalization_1/beta*
shape:*
allowed_devices
 *
dtype0*
_output_shapes
: *
	container 
?
;batch_normalization_1/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_1/beta*
_output_shapes
: 
?
!batch_normalization_1/beta/AssignAssignVariableOpbatch_normalization_1/beta,batch_normalization_1/beta/Initializer/zeros*
dtype0*
validate_shape( 
?
.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
dtype0*
_output_shapes
:
?
3batch_normalization_1/moving_mean/Initializer/zerosConst*
_output_shapes
:*
valueB*    *4
_class*
(&loc:@batch_normalization_1/moving_mean*
dtype0
?
!batch_normalization_1/moving_meanVarHandleOp*2
shared_name#!batch_normalization_1/moving_mean*
	container *
dtype0*
_output_shapes
: *
shape:*4
_class*
(&loc:@batch_normalization_1/moving_mean*
allowed_devices
 
?
Bbatch_normalization_1/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp!batch_normalization_1/moving_mean*
_output_shapes
: 
?
(batch_normalization_1/moving_mean/AssignAssignVariableOp!batch_normalization_1/moving_mean3batch_normalization_1/moving_mean/Initializer/zeros*
dtype0*
validate_shape( 
?
5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
dtype0*
_output_shapes
:
?
6batch_normalization_1/moving_variance/Initializer/onesConst*
valueB*  ??*
_output_shapes
:*8
_class.
,*loc:@batch_normalization_1/moving_variance*
dtype0
?
%batch_normalization_1/moving_varianceVarHandleOp*8
_class.
,*loc:@batch_normalization_1/moving_variance*6
shared_name'%batch_normalization_1/moving_variance*
shape:*
_output_shapes
: *
	container *
dtype0*
allowed_devices
 
?
Fbatch_normalization_1/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp%batch_normalization_1/moving_variance*
_output_shapes
: 
?
,batch_normalization_1/moving_variance/AssignAssignVariableOp%batch_normalization_1/moving_variance6batch_normalization_1/moving_variance/Initializer/ones*
dtype0*
validate_shape( 
?
9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
dtype0*
_output_shapes
:
?
batch_normalization_1/condIfkeras_learning_phasebatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_variance!stream_1/depthwise_conv2d/BiasAdd*@
_output_shapes.
,:<::: : : : : : : *
Tcond0
*6
then_branch'R%
#batch_normalization_1_cond_true_164*
Tin	
2*?
output_shapes.
,:<::: : : : : : : *&
_read_only_resource_inputs
*7
else_branch(R&
$batch_normalization_1_cond_false_165*
_lower_using_switch_merge(*
Tout
2

|
#batch_normalization_1/cond/IdentityIdentitybatch_normalization_1/cond*
T0*&
_output_shapes
:<
t
%batch_normalization_1/cond/Identity_1Identitybatch_normalization_1/cond:1*
T0*
_output_shapes
:
t
%batch_normalization_1/cond/Identity_2Identitybatch_normalization_1/cond:2*
T0*
_output_shapes
:
p
%batch_normalization_1/cond/Identity_3Identitybatch_normalization_1/cond:3*
_output_shapes
: *
T0
p
%batch_normalization_1/cond/Identity_4Identitybatch_normalization_1/cond:4*
T0*
_output_shapes
: 
p
%batch_normalization_1/cond/Identity_5Identitybatch_normalization_1/cond:5*
_output_shapes
: *
T0
p
%batch_normalization_1/cond/Identity_6Identitybatch_normalization_1/cond:6*
_output_shapes
: *
T0
p
%batch_normalization_1/cond/Identity_7Identitybatch_normalization_1/cond:7*
T0*
_output_shapes
: 
p
%batch_normalization_1/cond/Identity_8Identitybatch_normalization_1/cond:8*
_output_shapes
: *
T0
p
%batch_normalization_1/cond/Identity_9Identitybatch_normalization_1/cond:9*
_output_shapes
: *
T0
?
batch_normalization_1/cond_1StatelessIfkeras_learning_phase*
Tout
2* 
_read_only_resource_inputs
 *	
Tin
 *
output_shapes
: *9
else_branch*R(
&batch_normalization_1_cond_1_false_233*8
then_branch)R'
%batch_normalization_1_cond_1_true_232*
_lower_using_switch_merge(*
_output_shapes
: *
Tcond0

p
%batch_normalization_1/cond_1/IdentityIdentitybatch_normalization_1/cond_1*
T0*
_output_shapes
: 
?
+batch_normalization_1/AssignMovingAvg/sub/xConst*
valueB
 *  ??*
dtype0*4
_class*
(&loc:@batch_normalization_1/moving_mean*
_output_shapes
: 
?
)batch_normalization_1/AssignMovingAvg/subSub+batch_normalization_1/AssignMovingAvg/sub/x%batch_normalization_1/cond_1/Identity*4
_class*
(&loc:@batch_normalization_1/moving_mean*
T0*
_output_shapes
: 
?
4batch_normalization_1/AssignMovingAvg/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_output_shapes
:*
dtype0
?
+batch_normalization_1/AssignMovingAvg/sub_1Sub4batch_normalization_1/AssignMovingAvg/ReadVariableOp%batch_normalization_1/cond/Identity_1*
_output_shapes
:*4
_class*
(&loc:@batch_normalization_1/moving_mean*
T0
?
)batch_normalization_1/AssignMovingAvg/mulMul+batch_normalization_1/AssignMovingAvg/sub_1)batch_normalization_1/AssignMovingAvg/sub*4
_class*
(&loc:@batch_normalization_1/moving_mean*
T0*
_output_shapes
:
?
9batch_normalization_1/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp!batch_normalization_1/moving_mean)batch_normalization_1/AssignMovingAvg/mul*4
_class*
(&loc:@batch_normalization_1/moving_mean*
dtype0
?
6batch_normalization_1/AssignMovingAvg/ReadVariableOp_1ReadVariableOp!batch_normalization_1/moving_mean:^batch_normalization_1/AssignMovingAvg/AssignSubVariableOp*4
_class*
(&loc:@batch_normalization_1/moving_mean*
dtype0*
_output_shapes
:
?
-batch_normalization_1/AssignMovingAvg_1/sub/xConst*
dtype0*
valueB
 *  ??*8
_class.
,*loc:@batch_normalization_1/moving_variance*
_output_shapes
: 
?
+batch_normalization_1/AssignMovingAvg_1/subSub-batch_normalization_1/AssignMovingAvg_1/sub/x%batch_normalization_1/cond_1/Identity*
T0*8
_class.
,*loc:@batch_normalization_1/moving_variance*
_output_shapes
: 
?
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
dtype0*
_output_shapes
:
?
-batch_normalization_1/AssignMovingAvg_1/sub_1Sub6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp%batch_normalization_1/cond/Identity_2*
_output_shapes
:*
T0*8
_class.
,*loc:@batch_normalization_1/moving_variance
?
+batch_normalization_1/AssignMovingAvg_1/mulMul-batch_normalization_1/AssignMovingAvg_1/sub_1+batch_normalization_1/AssignMovingAvg_1/sub*
T0*
_output_shapes
:*8
_class.
,*loc:@batch_normalization_1/moving_variance
?
;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp%batch_normalization_1/moving_variance+batch_normalization_1/AssignMovingAvg_1/mul*8
_class.
,*loc:@batch_normalization_1/moving_variance*
dtype0
?
8batch_normalization_1/AssignMovingAvg_1/ReadVariableOp_1ReadVariableOp%batch_normalization_1/moving_variance<^batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOp*8
_class.
,*loc:@batch_normalization_1/moving_variance*
_output_shapes
:*
dtype0
o
activation_1/ReluRelu#batch_normalization_1/cond/Identity*
T0*&
_output_shapes
:<
?
:stream_2/depthwise_kernel/Initializer/random_uniform/shapeConst*%
valueB"            *
dtype0*,
_class"
 loc:@stream_2/depthwise_kernel*
_output_shapes
:
?
8stream_2/depthwise_kernel/Initializer/random_uniform/minConst*
valueB
 *?7?*
_output_shapes
: *,
_class"
 loc:@stream_2/depthwise_kernel*
dtype0
?
8stream_2/depthwise_kernel/Initializer/random_uniform/maxConst*
valueB
 *?7?*
_output_shapes
: *,
_class"
 loc:@stream_2/depthwise_kernel*
dtype0
?
Bstream_2/depthwise_kernel/Initializer/random_uniform/RandomUniformRandomUniform:stream_2/depthwise_kernel/Initializer/random_uniform/shape*

seed *&
_output_shapes
:*
dtype0*,
_class"
 loc:@stream_2/depthwise_kernel*
seed2 *
T0
?
8stream_2/depthwise_kernel/Initializer/random_uniform/subSub8stream_2/depthwise_kernel/Initializer/random_uniform/max8stream_2/depthwise_kernel/Initializer/random_uniform/min*
_output_shapes
: *,
_class"
 loc:@stream_2/depthwise_kernel*
T0
?
8stream_2/depthwise_kernel/Initializer/random_uniform/mulMulBstream_2/depthwise_kernel/Initializer/random_uniform/RandomUniform8stream_2/depthwise_kernel/Initializer/random_uniform/sub*
T0*,
_class"
 loc:@stream_2/depthwise_kernel*&
_output_shapes
:
?
4stream_2/depthwise_kernel/Initializer/random_uniformAddV28stream_2/depthwise_kernel/Initializer/random_uniform/mul8stream_2/depthwise_kernel/Initializer/random_uniform/min*,
_class"
 loc:@stream_2/depthwise_kernel*&
_output_shapes
:*
T0
?
stream_2/depthwise_kernelVarHandleOp**
shared_namestream_2/depthwise_kernel*,
_class"
 loc:@stream_2/depthwise_kernel*
	container *
shape:*
_output_shapes
: *
dtype0*
allowed_devices
 
?
:stream_2/depthwise_kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpstream_2/depthwise_kernel*
_output_shapes
: 
?
 stream_2/depthwise_kernel/AssignAssignVariableOpstream_2/depthwise_kernel4stream_2/depthwise_kernel/Initializer/random_uniform*
dtype0*
validate_shape( 
?
-stream_2/depthwise_kernel/Read/ReadVariableOpReadVariableOpstream_2/depthwise_kernel*&
_output_shapes
:*
dtype0
?
stream_2/bias/Initializer/zerosConst*
valueB*    *
_output_shapes
:* 
_class
loc:@stream_2/bias*
dtype0
?
stream_2/biasVarHandleOp*
shape:*
allowed_devices
 *
dtype0*
shared_namestream_2/bias*
	container *
_output_shapes
: * 
_class
loc:@stream_2/bias
k
.stream_2/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpstream_2/bias*
_output_shapes
: 
{
stream_2/bias/AssignAssignVariableOpstream_2/biasstream_2/bias/Initializer/zeros*
dtype0*
validate_shape( 
k
!stream_2/bias/Read/ReadVariableOpReadVariableOpstream_2/bias*
dtype0*
_output_shapes
:
?
4stream_2/depthwise_conv2d_1/depthwise/ReadVariableOpReadVariableOpstream_2/depthwise_kernel*&
_output_shapes
:*
dtype0
?
+stream_2/depthwise_conv2d_1/depthwise/ShapeConst*
_output_shapes
:*%
valueB"            *
dtype0
?
3stream_2/depthwise_conv2d_1/depthwise/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
?
%stream_2/depthwise_conv2d_1/depthwiseDepthwiseConv2dNativeactivation_1/Relu4stream_2/depthwise_conv2d_1/depthwise/ReadVariableOp*
T0*
	dilations
*&
_output_shapes
:9*
data_formatNHWC*
strides
*
paddingVALID*
explicit_paddings
 
|
2stream_2/depthwise_conv2d_1/BiasAdd/ReadVariableOpReadVariableOpstream_2/bias*
dtype0*
_output_shapes
:
?
#stream_2/depthwise_conv2d_1/BiasAddBiasAdd%stream_2/depthwise_conv2d_1/depthwise2stream_2/depthwise_conv2d_1/BiasAdd/ReadVariableOp*
data_formatNHWC*
T0*&
_output_shapes
:9
?
,batch_normalization_2/gamma/Initializer/onesConst*
_output_shapes
:*.
_class$
" loc:@batch_normalization_2/gamma*
valueB*  ??*
dtype0
?
batch_normalization_2/gammaVarHandleOp*,
shared_namebatch_normalization_2/gamma*
allowed_devices
 *
shape:*
_output_shapes
: *
	container *
dtype0*.
_class$
" loc:@batch_normalization_2/gamma
?
<batch_normalization_2/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_2/gamma*
_output_shapes
: 
?
"batch_normalization_2/gamma/AssignAssignVariableOpbatch_normalization_2/gamma,batch_normalization_2/gamma/Initializer/ones*
dtype0*
validate_shape( 
?
/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_2/gamma*
dtype0*
_output_shapes
:
?
,batch_normalization_2/beta/Initializer/zerosConst*-
_class#
!loc:@batch_normalization_2/beta*
dtype0*
_output_shapes
:*
valueB*    
?
batch_normalization_2/betaVarHandleOp*-
_class#
!loc:@batch_normalization_2/beta*
dtype0*
shape:*
_output_shapes
: *
	container *
allowed_devices
 *+
shared_namebatch_normalization_2/beta
?
;batch_normalization_2/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_2/beta*
_output_shapes
: 
?
!batch_normalization_2/beta/AssignAssignVariableOpbatch_normalization_2/beta,batch_normalization_2/beta/Initializer/zeros*
dtype0*
validate_shape( 
?
.batch_normalization_2/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_2/beta*
_output_shapes
:*
dtype0
?
3batch_normalization_2/moving_mean/Initializer/zerosConst*4
_class*
(&loc:@batch_normalization_2/moving_mean*
valueB*    *
_output_shapes
:*
dtype0
?
!batch_normalization_2/moving_meanVarHandleOp*
allowed_devices
 *
	container *2
shared_name#!batch_normalization_2/moving_mean*4
_class*
(&loc:@batch_normalization_2/moving_mean*
shape:*
_output_shapes
: *
dtype0
?
Bbatch_normalization_2/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp!batch_normalization_2/moving_mean*
_output_shapes
: 
?
(batch_normalization_2/moving_mean/AssignAssignVariableOp!batch_normalization_2/moving_mean3batch_normalization_2/moving_mean/Initializer/zeros*
dtype0*
validate_shape( 
?
5batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*
_output_shapes
:*
dtype0
?
6batch_normalization_2/moving_variance/Initializer/onesConst*8
_class.
,*loc:@batch_normalization_2/moving_variance*
_output_shapes
:*
dtype0*
valueB*  ??
?
%batch_normalization_2/moving_varianceVarHandleOp*
allowed_devices
 *
dtype0*6
shared_name'%batch_normalization_2/moving_variance*
_output_shapes
: *
	container *8
_class.
,*loc:@batch_normalization_2/moving_variance*
shape:
?
Fbatch_normalization_2/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp%batch_normalization_2/moving_variance*
_output_shapes
: 
?
,batch_normalization_2/moving_variance/AssignAssignVariableOp%batch_normalization_2/moving_variance6batch_normalization_2/moving_variance/Initializer/ones*
dtype0*
validate_shape( 
?
9batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_2/moving_variance*
_output_shapes
:*
dtype0
?
batch_normalization_2/condIfkeras_learning_phasebatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_variance#stream_2/depthwise_conv2d_1/BiasAdd*
Tin	
2*@
_output_shapes.
,:9::: : : : : : : *?
output_shapes.
,:9::: : : : : : : *7
else_branch(R&
$batch_normalization_2_cond_false_290*
_lower_using_switch_merge(*
Tcond0
*
Tout
2
*&
_read_only_resource_inputs
*6
then_branch'R%
#batch_normalization_2_cond_true_289
|
#batch_normalization_2/cond/IdentityIdentitybatch_normalization_2/cond*
T0*&
_output_shapes
:9
t
%batch_normalization_2/cond/Identity_1Identitybatch_normalization_2/cond:1*
T0*
_output_shapes
:
t
%batch_normalization_2/cond/Identity_2Identitybatch_normalization_2/cond:2*
T0*
_output_shapes
:
p
%batch_normalization_2/cond/Identity_3Identitybatch_normalization_2/cond:3*
_output_shapes
: *
T0
p
%batch_normalization_2/cond/Identity_4Identitybatch_normalization_2/cond:4*
_output_shapes
: *
T0
p
%batch_normalization_2/cond/Identity_5Identitybatch_normalization_2/cond:5*
T0*
_output_shapes
: 
p
%batch_normalization_2/cond/Identity_6Identitybatch_normalization_2/cond:6*
T0*
_output_shapes
: 
p
%batch_normalization_2/cond/Identity_7Identitybatch_normalization_2/cond:7*
T0*
_output_shapes
: 
p
%batch_normalization_2/cond/Identity_8Identitybatch_normalization_2/cond:8*
_output_shapes
: *
T0
p
%batch_normalization_2/cond/Identity_9Identitybatch_normalization_2/cond:9*
_output_shapes
: *
T0
?
batch_normalization_2/cond_1StatelessIfkeras_learning_phase*	
Tin
 *9
else_branch*R(
&batch_normalization_2_cond_1_false_358*8
then_branch)R'
%batch_normalization_2_cond_1_true_357*
output_shapes
: *
_output_shapes
: * 
_read_only_resource_inputs
 *
Tcond0
*
Tout
2*
_lower_using_switch_merge(
p
%batch_normalization_2/cond_1/IdentityIdentitybatch_normalization_2/cond_1*
T0*
_output_shapes
: 
?
+batch_normalization_2/AssignMovingAvg/sub/xConst*
valueB
 *  ??*
_output_shapes
: *4
_class*
(&loc:@batch_normalization_2/moving_mean*
dtype0
?
)batch_normalization_2/AssignMovingAvg/subSub+batch_normalization_2/AssignMovingAvg/sub/x%batch_normalization_2/cond_1/Identity*
_output_shapes
: *4
_class*
(&loc:@batch_normalization_2/moving_mean*
T0
?
4batch_normalization_2/AssignMovingAvg/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*
dtype0*
_output_shapes
:
?
+batch_normalization_2/AssignMovingAvg/sub_1Sub4batch_normalization_2/AssignMovingAvg/ReadVariableOp%batch_normalization_2/cond/Identity_1*
_output_shapes
:*
T0*4
_class*
(&loc:@batch_normalization_2/moving_mean
?
)batch_normalization_2/AssignMovingAvg/mulMul+batch_normalization_2/AssignMovingAvg/sub_1)batch_normalization_2/AssignMovingAvg/sub*
T0*4
_class*
(&loc:@batch_normalization_2/moving_mean*
_output_shapes
:
?
9batch_normalization_2/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp!batch_normalization_2/moving_mean)batch_normalization_2/AssignMovingAvg/mul*4
_class*
(&loc:@batch_normalization_2/moving_mean*
dtype0
?
6batch_normalization_2/AssignMovingAvg/ReadVariableOp_1ReadVariableOp!batch_normalization_2/moving_mean:^batch_normalization_2/AssignMovingAvg/AssignSubVariableOp*
dtype0*
_output_shapes
:*4
_class*
(&loc:@batch_normalization_2/moving_mean
?
-batch_normalization_2/AssignMovingAvg_1/sub/xConst*
_output_shapes
: *
dtype0*8
_class.
,*loc:@batch_normalization_2/moving_variance*
valueB
 *  ??
?
+batch_normalization_2/AssignMovingAvg_1/subSub-batch_normalization_2/AssignMovingAvg_1/sub/x%batch_normalization_2/cond_1/Identity*8
_class.
,*loc:@batch_normalization_2/moving_variance*
_output_shapes
: *
T0
?
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpReadVariableOp%batch_normalization_2/moving_variance*
_output_shapes
:*
dtype0
?
-batch_normalization_2/AssignMovingAvg_1/sub_1Sub6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp%batch_normalization_2/cond/Identity_2*
_output_shapes
:*
T0*8
_class.
,*loc:@batch_normalization_2/moving_variance
?
+batch_normalization_2/AssignMovingAvg_1/mulMul-batch_normalization_2/AssignMovingAvg_1/sub_1+batch_normalization_2/AssignMovingAvg_1/sub*
T0*
_output_shapes
:*8
_class.
,*loc:@batch_normalization_2/moving_variance
?
;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp%batch_normalization_2/moving_variance+batch_normalization_2/AssignMovingAvg_1/mul*8
_class.
,*loc:@batch_normalization_2/moving_variance*
dtype0
?
8batch_normalization_2/AssignMovingAvg_1/ReadVariableOp_1ReadVariableOp%batch_normalization_2/moving_variance<^batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOp*
dtype0*8
_class.
,*loc:@batch_normalization_2/moving_variance*
_output_shapes
:
o
activation_2/ReluRelu#batch_normalization_2/cond/Identity*
T0*&
_output_shapes
:9
?
"stream_3/average_pooling2d/AvgPoolAvgPoolactivation_2/Relu*
paddingVALID*
data_formatNHWC*&
_output_shapes
:*
T0*
strides
*
ksize

?
:stream_4/depthwise_kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*%
valueB"            *
dtype0*,
_class"
 loc:@stream_4/depthwise_kernel
?
8stream_4/depthwise_kernel/Initializer/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *׳ݾ*,
_class"
 loc:@stream_4/depthwise_kernel
?
8stream_4/depthwise_kernel/Initializer/random_uniform/maxConst*
dtype0*,
_class"
 loc:@stream_4/depthwise_kernel*
valueB
 *׳?>*
_output_shapes
: 
?
Bstream_4/depthwise_kernel/Initializer/random_uniform/RandomUniformRandomUniform:stream_4/depthwise_kernel/Initializer/random_uniform/shape*&
_output_shapes
:*
dtype0*

seed *,
_class"
 loc:@stream_4/depthwise_kernel*
T0*
seed2 
?
8stream_4/depthwise_kernel/Initializer/random_uniform/subSub8stream_4/depthwise_kernel/Initializer/random_uniform/max8stream_4/depthwise_kernel/Initializer/random_uniform/min*,
_class"
 loc:@stream_4/depthwise_kernel*
T0*
_output_shapes
: 
?
8stream_4/depthwise_kernel/Initializer/random_uniform/mulMulBstream_4/depthwise_kernel/Initializer/random_uniform/RandomUniform8stream_4/depthwise_kernel/Initializer/random_uniform/sub*&
_output_shapes
:*
T0*,
_class"
 loc:@stream_4/depthwise_kernel
?
4stream_4/depthwise_kernel/Initializer/random_uniformAddV28stream_4/depthwise_kernel/Initializer/random_uniform/mul8stream_4/depthwise_kernel/Initializer/random_uniform/min*,
_class"
 loc:@stream_4/depthwise_kernel*
T0*&
_output_shapes
:
?
stream_4/depthwise_kernelVarHandleOp*
shape:*
allowed_devices
 *,
_class"
 loc:@stream_4/depthwise_kernel*
dtype0*
	container **
shared_namestream_4/depthwise_kernel*
_output_shapes
: 
?
:stream_4/depthwise_kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpstream_4/depthwise_kernel*
_output_shapes
: 
?
 stream_4/depthwise_kernel/AssignAssignVariableOpstream_4/depthwise_kernel4stream_4/depthwise_kernel/Initializer/random_uniform*
dtype0*
validate_shape( 
?
-stream_4/depthwise_kernel/Read/ReadVariableOpReadVariableOpstream_4/depthwise_kernel*&
_output_shapes
:*
dtype0
?
stream_4/bias/Initializer/zerosConst*
dtype0*
_output_shapes
:* 
_class
loc:@stream_4/bias*
valueB*    
?
stream_4/biasVarHandleOp*
shape:*
allowed_devices
 *
	container * 
_class
loc:@stream_4/bias*
shared_namestream_4/bias*
dtype0*
_output_shapes
: 
k
.stream_4/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpstream_4/bias*
_output_shapes
: 
{
stream_4/bias/AssignAssignVariableOpstream_4/biasstream_4/bias/Initializer/zeros*
dtype0*
validate_shape( 
k
!stream_4/bias/Read/ReadVariableOpReadVariableOpstream_4/bias*
dtype0*
_output_shapes
:
?
4stream_4/depthwise_conv2d_2/depthwise/ReadVariableOpReadVariableOpstream_4/depthwise_kernel*&
_output_shapes
:*
dtype0
?
+stream_4/depthwise_conv2d_2/depthwise/ShapeConst*
dtype0*%
valueB"            *
_output_shapes
:
?
3stream_4/depthwise_conv2d_2/depthwise/dilation_rateConst*
valueB"      *
_output_shapes
:*
dtype0
?
%stream_4/depthwise_conv2d_2/depthwiseDepthwiseConv2dNative"stream_3/average_pooling2d/AvgPool4stream_4/depthwise_conv2d_2/depthwise/ReadVariableOp*
	dilations
*
data_formatNHWC*&
_output_shapes
:*
T0*
paddingVALID*
explicit_paddings
 *
strides

|
2stream_4/depthwise_conv2d_2/BiasAdd/ReadVariableOpReadVariableOpstream_4/bias*
_output_shapes
:*
dtype0
?
#stream_4/depthwise_conv2d_2/BiasAddBiasAdd%stream_4/depthwise_conv2d_2/depthwise2stream_4/depthwise_conv2d_2/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC*&
_output_shapes
:
?
,batch_normalization_3/gamma/Initializer/onesConst*
_output_shapes
:*
dtype0*.
_class$
" loc:@batch_normalization_3/gamma*
valueB*  ??
?
batch_normalization_3/gammaVarHandleOp*.
_class$
" loc:@batch_normalization_3/gamma*,
shared_namebatch_normalization_3/gamma*
	container *
_output_shapes
: *
allowed_devices
 *
shape:*
dtype0
?
<batch_normalization_3/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_3/gamma*
_output_shapes
: 
?
"batch_normalization_3/gamma/AssignAssignVariableOpbatch_normalization_3/gamma,batch_normalization_3/gamma/Initializer/ones*
dtype0*
validate_shape( 
?
/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_3/gamma*
_output_shapes
:*
dtype0
?
,batch_normalization_3/beta/Initializer/zerosConst*
_output_shapes
:*-
_class#
!loc:@batch_normalization_3/beta*
dtype0*
valueB*    
?
batch_normalization_3/betaVarHandleOp*
_output_shapes
: *
shape:*-
_class#
!loc:@batch_normalization_3/beta*
allowed_devices
 *
	container *
dtype0*+
shared_namebatch_normalization_3/beta
?
;batch_normalization_3/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_3/beta*
_output_shapes
: 
?
!batch_normalization_3/beta/AssignAssignVariableOpbatch_normalization_3/beta,batch_normalization_3/beta/Initializer/zeros*
dtype0*
validate_shape( 
?
.batch_normalization_3/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_3/beta*
_output_shapes
:*
dtype0
?
3batch_normalization_3/moving_mean/Initializer/zerosConst*
dtype0*
_output_shapes
:*4
_class*
(&loc:@batch_normalization_3/moving_mean*
valueB*    
?
!batch_normalization_3/moving_meanVarHandleOp*
allowed_devices
 *2
shared_name#!batch_normalization_3/moving_mean*
shape:*4
_class*
(&loc:@batch_normalization_3/moving_mean*
_output_shapes
: *
	container *
dtype0
?
Bbatch_normalization_3/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp!batch_normalization_3/moving_mean*
_output_shapes
: 
?
(batch_normalization_3/moving_mean/AssignAssignVariableOp!batch_normalization_3/moving_mean3batch_normalization_3/moving_mean/Initializer/zeros*
dtype0*
validate_shape( 
?
5batch_normalization_3/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_3/moving_mean*
dtype0*
_output_shapes
:
?
6batch_normalization_3/moving_variance/Initializer/onesConst*
dtype0*
valueB*  ??*8
_class.
,*loc:@batch_normalization_3/moving_variance*
_output_shapes
:
?
%batch_normalization_3/moving_varianceVarHandleOp*
	container *
_output_shapes
: *
allowed_devices
 *
dtype0*
shape:*8
_class.
,*loc:@batch_normalization_3/moving_variance*6
shared_name'%batch_normalization_3/moving_variance
?
Fbatch_normalization_3/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp%batch_normalization_3/moving_variance*
_output_shapes
: 
?
,batch_normalization_3/moving_variance/AssignAssignVariableOp%batch_normalization_3/moving_variance6batch_normalization_3/moving_variance/Initializer/ones*
dtype0*
validate_shape( 
?
9batch_normalization_3/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_3/moving_variance*
dtype0*
_output_shapes
:
?
batch_normalization_3/condIfkeras_learning_phasebatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_variance#stream_4/depthwise_conv2d_2/BiasAdd*6
then_branch'R%
#batch_normalization_3_cond_true_415*
_lower_using_switch_merge(*
Tin	
2*&
_read_only_resource_inputs
*7
else_branch(R&
$batch_normalization_3_cond_false_416*?
output_shapes.
,:::: : : : : : : *
Tout
2
*
Tcond0
*@
_output_shapes.
,:::: : : : : : : 
|
#batch_normalization_3/cond/IdentityIdentitybatch_normalization_3/cond*
T0*&
_output_shapes
:
t
%batch_normalization_3/cond/Identity_1Identitybatch_normalization_3/cond:1*
T0*
_output_shapes
:
t
%batch_normalization_3/cond/Identity_2Identitybatch_normalization_3/cond:2*
_output_shapes
:*
T0
p
%batch_normalization_3/cond/Identity_3Identitybatch_normalization_3/cond:3*
_output_shapes
: *
T0
p
%batch_normalization_3/cond/Identity_4Identitybatch_normalization_3/cond:4*
T0*
_output_shapes
: 
p
%batch_normalization_3/cond/Identity_5Identitybatch_normalization_3/cond:5*
T0*
_output_shapes
: 
p
%batch_normalization_3/cond/Identity_6Identitybatch_normalization_3/cond:6*
T0*
_output_shapes
: 
p
%batch_normalization_3/cond/Identity_7Identitybatch_normalization_3/cond:7*
T0*
_output_shapes
: 
p
%batch_normalization_3/cond/Identity_8Identitybatch_normalization_3/cond:8*
_output_shapes
: *
T0
p
%batch_normalization_3/cond/Identity_9Identitybatch_normalization_3/cond:9*
T0*
_output_shapes
: 
?
batch_normalization_3/cond_1StatelessIfkeras_learning_phase*
Tcond0
*
_lower_using_switch_merge(*
Tout
2*	
Tin
 *9
else_branch*R(
&batch_normalization_3_cond_1_false_484*
output_shapes
: *8
then_branch)R'
%batch_normalization_3_cond_1_true_483* 
_read_only_resource_inputs
 *
_output_shapes
: 
p
%batch_normalization_3/cond_1/IdentityIdentitybatch_normalization_3/cond_1*
_output_shapes
: *
T0
?
+batch_normalization_3/AssignMovingAvg/sub/xConst*
_output_shapes
: *
dtype0*4
_class*
(&loc:@batch_normalization_3/moving_mean*
valueB
 *  ??
?
)batch_normalization_3/AssignMovingAvg/subSub+batch_normalization_3/AssignMovingAvg/sub/x%batch_normalization_3/cond_1/Identity*4
_class*
(&loc:@batch_normalization_3/moving_mean*
_output_shapes
: *
T0
?
4batch_normalization_3/AssignMovingAvg/ReadVariableOpReadVariableOp!batch_normalization_3/moving_mean*
dtype0*
_output_shapes
:
?
+batch_normalization_3/AssignMovingAvg/sub_1Sub4batch_normalization_3/AssignMovingAvg/ReadVariableOp%batch_normalization_3/cond/Identity_1*
_output_shapes
:*
T0*4
_class*
(&loc:@batch_normalization_3/moving_mean
?
)batch_normalization_3/AssignMovingAvg/mulMul+batch_normalization_3/AssignMovingAvg/sub_1)batch_normalization_3/AssignMovingAvg/sub*4
_class*
(&loc:@batch_normalization_3/moving_mean*
T0*
_output_shapes
:
?
9batch_normalization_3/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp!batch_normalization_3/moving_mean)batch_normalization_3/AssignMovingAvg/mul*4
_class*
(&loc:@batch_normalization_3/moving_mean*
dtype0
?
6batch_normalization_3/AssignMovingAvg/ReadVariableOp_1ReadVariableOp!batch_normalization_3/moving_mean:^batch_normalization_3/AssignMovingAvg/AssignSubVariableOp*
_output_shapes
:*4
_class*
(&loc:@batch_normalization_3/moving_mean*
dtype0
?
-batch_normalization_3/AssignMovingAvg_1/sub/xConst*8
_class.
,*loc:@batch_normalization_3/moving_variance*
valueB
 *  ??*
_output_shapes
: *
dtype0
?
+batch_normalization_3/AssignMovingAvg_1/subSub-batch_normalization_3/AssignMovingAvg_1/sub/x%batch_normalization_3/cond_1/Identity*
_output_shapes
: *
T0*8
_class.
,*loc:@batch_normalization_3/moving_variance
?
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOpReadVariableOp%batch_normalization_3/moving_variance*
dtype0*
_output_shapes
:
?
-batch_normalization_3/AssignMovingAvg_1/sub_1Sub6batch_normalization_3/AssignMovingAvg_1/ReadVariableOp%batch_normalization_3/cond/Identity_2*
_output_shapes
:*
T0*8
_class.
,*loc:@batch_normalization_3/moving_variance
?
+batch_normalization_3/AssignMovingAvg_1/mulMul-batch_normalization_3/AssignMovingAvg_1/sub_1+batch_normalization_3/AssignMovingAvg_1/sub*
_output_shapes
:*8
_class.
,*loc:@batch_normalization_3/moving_variance*
T0
?
;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp%batch_normalization_3/moving_variance+batch_normalization_3/AssignMovingAvg_1/mul*
dtype0*8
_class.
,*loc:@batch_normalization_3/moving_variance
?
8batch_normalization_3/AssignMovingAvg_1/ReadVariableOp_1ReadVariableOp%batch_normalization_3/moving_variance<^batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOp*8
_class.
,*loc:@batch_normalization_3/moving_variance*
_output_shapes
:*
dtype0
o
activation_3/ReluRelu#batch_normalization_3/cond/Identity*
T0*&
_output_shapes
:
?
:stream_5/depthwise_kernel/Initializer/random_uniform/shapeConst*,
_class"
 loc:@stream_5/depthwise_kernel*
dtype0*%
valueB"            *
_output_shapes
:
?
8stream_5/depthwise_kernel/Initializer/random_uniform/minConst*
_output_shapes
: *
valueB
 *b???*,
_class"
 loc:@stream_5/depthwise_kernel*
dtype0
?
8stream_5/depthwise_kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *b??>*,
_class"
 loc:@stream_5/depthwise_kernel
?
Bstream_5/depthwise_kernel/Initializer/random_uniform/RandomUniformRandomUniform:stream_5/depthwise_kernel/Initializer/random_uniform/shape*
seed2 *

seed *
dtype0*&
_output_shapes
:*
T0*,
_class"
 loc:@stream_5/depthwise_kernel
?
8stream_5/depthwise_kernel/Initializer/random_uniform/subSub8stream_5/depthwise_kernel/Initializer/random_uniform/max8stream_5/depthwise_kernel/Initializer/random_uniform/min*
T0*
_output_shapes
: *,
_class"
 loc:@stream_5/depthwise_kernel
?
8stream_5/depthwise_kernel/Initializer/random_uniform/mulMulBstream_5/depthwise_kernel/Initializer/random_uniform/RandomUniform8stream_5/depthwise_kernel/Initializer/random_uniform/sub*
T0*&
_output_shapes
:*,
_class"
 loc:@stream_5/depthwise_kernel
?
4stream_5/depthwise_kernel/Initializer/random_uniformAddV28stream_5/depthwise_kernel/Initializer/random_uniform/mul8stream_5/depthwise_kernel/Initializer/random_uniform/min*,
_class"
 loc:@stream_5/depthwise_kernel*&
_output_shapes
:*
T0
?
stream_5/depthwise_kernelVarHandleOp*
shape:*
_output_shapes
: *,
_class"
 loc:@stream_5/depthwise_kernel**
shared_namestream_5/depthwise_kernel*
dtype0*
	container *
allowed_devices
 
?
:stream_5/depthwise_kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpstream_5/depthwise_kernel*
_output_shapes
: 
?
 stream_5/depthwise_kernel/AssignAssignVariableOpstream_5/depthwise_kernel4stream_5/depthwise_kernel/Initializer/random_uniform*
dtype0*
validate_shape( 
?
-stream_5/depthwise_kernel/Read/ReadVariableOpReadVariableOpstream_5/depthwise_kernel*&
_output_shapes
:*
dtype0
?
stream_5/bias/Initializer/zerosConst*
_output_shapes
:*
valueB*    * 
_class
loc:@stream_5/bias*
dtype0
?
stream_5/biasVarHandleOp*
	container * 
_class
loc:@stream_5/bias*
shared_namestream_5/bias*
_output_shapes
: *
dtype0*
shape:*
allowed_devices
 
k
.stream_5/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpstream_5/bias*
_output_shapes
: 
{
stream_5/bias/AssignAssignVariableOpstream_5/biasstream_5/bias/Initializer/zeros*
dtype0*
validate_shape( 
k
!stream_5/bias/Read/ReadVariableOpReadVariableOpstream_5/bias*
_output_shapes
:*
dtype0
?
4stream_5/depthwise_conv2d_3/depthwise/ReadVariableOpReadVariableOpstream_5/depthwise_kernel*&
_output_shapes
:*
dtype0
?
+stream_5/depthwise_conv2d_3/depthwise/ShapeConst*
dtype0*%
valueB"            *
_output_shapes
:
?
3stream_5/depthwise_conv2d_3/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
?
%stream_5/depthwise_conv2d_3/depthwiseDepthwiseConv2dNativeactivation_3/Relu4stream_5/depthwise_conv2d_3/depthwise/ReadVariableOp*
data_formatNHWC*
T0*&
_output_shapes
:*
explicit_paddings
 *
	dilations
*
strides
*
paddingVALID
|
2stream_5/depthwise_conv2d_3/BiasAdd/ReadVariableOpReadVariableOpstream_5/bias*
_output_shapes
:*
dtype0
?
#stream_5/depthwise_conv2d_3/BiasAddBiasAdd%stream_5/depthwise_conv2d_3/depthwise2stream_5/depthwise_conv2d_3/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC*&
_output_shapes
:
?
,batch_normalization_4/gamma/Initializer/onesConst*.
_class$
" loc:@batch_normalization_4/gamma*
_output_shapes
:*
valueB*  ??*
dtype0
?
batch_normalization_4/gammaVarHandleOp*.
_class$
" loc:@batch_normalization_4/gamma*
allowed_devices
 *
	container *
shape:*
dtype0*
_output_shapes
: *,
shared_namebatch_normalization_4/gamma
?
<batch_normalization_4/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_4/gamma*
_output_shapes
: 
?
"batch_normalization_4/gamma/AssignAssignVariableOpbatch_normalization_4/gamma,batch_normalization_4/gamma/Initializer/ones*
dtype0*
validate_shape( 
?
/batch_normalization_4/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_4/gamma*
_output_shapes
:*
dtype0
?
,batch_normalization_4/beta/Initializer/zerosConst*
_output_shapes
:*
dtype0*
valueB*    *-
_class#
!loc:@batch_normalization_4/beta
?
batch_normalization_4/betaVarHandleOp*-
_class#
!loc:@batch_normalization_4/beta*
dtype0*+
shared_namebatch_normalization_4/beta*
shape:*
	container *
_output_shapes
: *
allowed_devices
 
?
;batch_normalization_4/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_4/beta*
_output_shapes
: 
?
!batch_normalization_4/beta/AssignAssignVariableOpbatch_normalization_4/beta,batch_normalization_4/beta/Initializer/zeros*
dtype0*
validate_shape( 
?
.batch_normalization_4/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_4/beta*
_output_shapes
:*
dtype0
?
3batch_normalization_4/moving_mean/Initializer/zerosConst*4
_class*
(&loc:@batch_normalization_4/moving_mean*
valueB*    *
_output_shapes
:*
dtype0
?
!batch_normalization_4/moving_meanVarHandleOp*
_output_shapes
: *2
shared_name#!batch_normalization_4/moving_mean*
dtype0*
shape:*
allowed_devices
 *
	container *4
_class*
(&loc:@batch_normalization_4/moving_mean
?
Bbatch_normalization_4/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp!batch_normalization_4/moving_mean*
_output_shapes
: 
?
(batch_normalization_4/moving_mean/AssignAssignVariableOp!batch_normalization_4/moving_mean3batch_normalization_4/moving_mean/Initializer/zeros*
dtype0*
validate_shape( 
?
5batch_normalization_4/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_4/moving_mean*
_output_shapes
:*
dtype0
?
6batch_normalization_4/moving_variance/Initializer/onesConst*
dtype0*
_output_shapes
:*8
_class.
,*loc:@batch_normalization_4/moving_variance*
valueB*  ??
?
%batch_normalization_4/moving_varianceVarHandleOp*8
_class.
,*loc:@batch_normalization_4/moving_variance*
allowed_devices
 *
shape:*
dtype0*
_output_shapes
: *6
shared_name'%batch_normalization_4/moving_variance*
	container 
?
Fbatch_normalization_4/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp%batch_normalization_4/moving_variance*
_output_shapes
: 
?
,batch_normalization_4/moving_variance/AssignAssignVariableOp%batch_normalization_4/moving_variance6batch_normalization_4/moving_variance/Initializer/ones*
dtype0*
validate_shape( 
?
9batch_normalization_4/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_4/moving_variance*
dtype0*
_output_shapes
:
?
batch_normalization_4/condIfkeras_learning_phasebatch_normalization_4/gammabatch_normalization_4/beta!batch_normalization_4/moving_mean%batch_normalization_4/moving_variance#stream_5/depthwise_conv2d_3/BiasAdd*6
then_branch'R%
#batch_normalization_4_cond_true_540*
_lower_using_switch_merge(*7
else_branch(R&
$batch_normalization_4_cond_false_541*?
output_shapes.
,:::: : : : : : : *
Tcond0
*&
_read_only_resource_inputs
*
Tout
2
*
Tin	
2*@
_output_shapes.
,:::: : : : : : : 
|
#batch_normalization_4/cond/IdentityIdentitybatch_normalization_4/cond*
T0*&
_output_shapes
:
t
%batch_normalization_4/cond/Identity_1Identitybatch_normalization_4/cond:1*
T0*
_output_shapes
:
t
%batch_normalization_4/cond/Identity_2Identitybatch_normalization_4/cond:2*
T0*
_output_shapes
:
p
%batch_normalization_4/cond/Identity_3Identitybatch_normalization_4/cond:3*
_output_shapes
: *
T0
p
%batch_normalization_4/cond/Identity_4Identitybatch_normalization_4/cond:4*
T0*
_output_shapes
: 
p
%batch_normalization_4/cond/Identity_5Identitybatch_normalization_4/cond:5*
T0*
_output_shapes
: 
p
%batch_normalization_4/cond/Identity_6Identitybatch_normalization_4/cond:6*
T0*
_output_shapes
: 
p
%batch_normalization_4/cond/Identity_7Identitybatch_normalization_4/cond:7*
T0*
_output_shapes
: 
p
%batch_normalization_4/cond/Identity_8Identitybatch_normalization_4/cond:8*
_output_shapes
: *
T0
p
%batch_normalization_4/cond/Identity_9Identitybatch_normalization_4/cond:9*
_output_shapes
: *
T0
?
batch_normalization_4/cond_1StatelessIfkeras_learning_phase*
output_shapes
: *
_output_shapes
: * 
_read_only_resource_inputs
 *
Tcond0
*9
else_branch*R(
&batch_normalization_4_cond_1_false_609*
Tout
2*
_lower_using_switch_merge(*	
Tin
 *8
then_branch)R'
%batch_normalization_4_cond_1_true_608
p
%batch_normalization_4/cond_1/IdentityIdentitybatch_normalization_4/cond_1*
T0*
_output_shapes
: 
?
+batch_normalization_4/AssignMovingAvg/sub/xConst*4
_class*
(&loc:@batch_normalization_4/moving_mean*
_output_shapes
: *
valueB
 *  ??*
dtype0
?
)batch_normalization_4/AssignMovingAvg/subSub+batch_normalization_4/AssignMovingAvg/sub/x%batch_normalization_4/cond_1/Identity*
_output_shapes
: *
T0*4
_class*
(&loc:@batch_normalization_4/moving_mean
?
4batch_normalization_4/AssignMovingAvg/ReadVariableOpReadVariableOp!batch_normalization_4/moving_mean*
_output_shapes
:*
dtype0
?
+batch_normalization_4/AssignMovingAvg/sub_1Sub4batch_normalization_4/AssignMovingAvg/ReadVariableOp%batch_normalization_4/cond/Identity_1*
_output_shapes
:*4
_class*
(&loc:@batch_normalization_4/moving_mean*
T0
?
)batch_normalization_4/AssignMovingAvg/mulMul+batch_normalization_4/AssignMovingAvg/sub_1)batch_normalization_4/AssignMovingAvg/sub*4
_class*
(&loc:@batch_normalization_4/moving_mean*
_output_shapes
:*
T0
?
9batch_normalization_4/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp!batch_normalization_4/moving_mean)batch_normalization_4/AssignMovingAvg/mul*4
_class*
(&loc:@batch_normalization_4/moving_mean*
dtype0
?
6batch_normalization_4/AssignMovingAvg/ReadVariableOp_1ReadVariableOp!batch_normalization_4/moving_mean:^batch_normalization_4/AssignMovingAvg/AssignSubVariableOp*
_output_shapes
:*
dtype0*4
_class*
(&loc:@batch_normalization_4/moving_mean
?
-batch_normalization_4/AssignMovingAvg_1/sub/xConst*8
_class.
,*loc:@batch_normalization_4/moving_variance*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
+batch_normalization_4/AssignMovingAvg_1/subSub-batch_normalization_4/AssignMovingAvg_1/sub/x%batch_normalization_4/cond_1/Identity*
T0*
_output_shapes
: *8
_class.
,*loc:@batch_normalization_4/moving_variance
?
6batch_normalization_4/AssignMovingAvg_1/ReadVariableOpReadVariableOp%batch_normalization_4/moving_variance*
_output_shapes
:*
dtype0
?
-batch_normalization_4/AssignMovingAvg_1/sub_1Sub6batch_normalization_4/AssignMovingAvg_1/ReadVariableOp%batch_normalization_4/cond/Identity_2*
T0*8
_class.
,*loc:@batch_normalization_4/moving_variance*
_output_shapes
:
?
+batch_normalization_4/AssignMovingAvg_1/mulMul-batch_normalization_4/AssignMovingAvg_1/sub_1+batch_normalization_4/AssignMovingAvg_1/sub*8
_class.
,*loc:@batch_normalization_4/moving_variance*
T0*
_output_shapes
:
?
;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp%batch_normalization_4/moving_variance+batch_normalization_4/AssignMovingAvg_1/mul*8
_class.
,*loc:@batch_normalization_4/moving_variance*
dtype0
?
8batch_normalization_4/AssignMovingAvg_1/ReadVariableOp_1ReadVariableOp%batch_normalization_4/moving_variance<^batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOp*8
_class.
,*loc:@batch_normalization_4/moving_variance*
dtype0*
_output_shapes
:
o
activation_4/ReluRelu#batch_normalization_4/cond/Identity*
T0*&
_output_shapes
:
?
dropout/condIfkeras_learning_phaseactivation_4/Relu* 
_read_only_resource_inputs
 *
Tin
2*)
else_branchR
dropout_cond_false_630*
Tout

2*(
then_branchR
dropout_cond_true_629*3
output_shapes"
 :: : : : : : : *
_lower_using_switch_merge(*
Tcond0
*4
_output_shapes"
 :: : : : : : : 
`
dropout/cond/IdentityIdentitydropout/cond*
T0*&
_output_shapes
:
T
dropout/cond/Identity_1Identitydropout/cond:1*
T0*
_output_shapes
: 
T
dropout/cond/Identity_2Identitydropout/cond:2*
T0*
_output_shapes
: 
T
dropout/cond/Identity_3Identitydropout/cond:3*
_output_shapes
: *
T0
T
dropout/cond/Identity_4Identitydropout/cond:4*
_output_shapes
: *
T0
T
dropout/cond/Identity_5Identitydropout/cond:5*
_output_shapes
: *
T0
T
dropout/cond/Identity_6Identitydropout/cond:6*
T0*
_output_shapes
: 
T
dropout/cond/Identity_7Identitydropout/cond:7*
_output_shapes
: *
T0
g
stream_6/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????0   
?
stream_6/flatten/ReshapeReshapedropout/cond/Identitystream_6/flatten/Const*
Tshape0*
T0*
_output_shapes

:0
?
-dense/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*
dtype0*
valueB"0   
   *
_class
loc:@dense/kernel
?
+dense/kernel/Initializer/random_uniform/minConst*
dtype0*
valueB
 *.???*
_class
loc:@dense/kernel*
_output_shapes
: 
?
+dense/kernel/Initializer/random_uniform/maxConst*
valueB
 *.??>*
dtype0*
_output_shapes
: *
_class
loc:@dense/kernel
?
5dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform-dense/kernel/Initializer/random_uniform/shape*
_output_shapes

:0
*
dtype0*
_class
loc:@dense/kernel*

seed *
seed2 *
T0
?
+dense/kernel/Initializer/random_uniform/subSub+dense/kernel/Initializer/random_uniform/max+dense/kernel/Initializer/random_uniform/min*
_class
loc:@dense/kernel*
_output_shapes
: *
T0
?
+dense/kernel/Initializer/random_uniform/mulMul5dense/kernel/Initializer/random_uniform/RandomUniform+dense/kernel/Initializer/random_uniform/sub*
_class
loc:@dense/kernel*
T0*
_output_shapes

:0

?
'dense/kernel/Initializer/random_uniformAddV2+dense/kernel/Initializer/random_uniform/mul+dense/kernel/Initializer/random_uniform/min*
_output_shapes

:0
*
T0*
_class
loc:@dense/kernel
?
dense/kernelVarHandleOp*
shared_namedense/kernel*
dtype0*
_class
loc:@dense/kernel*
_output_shapes
: *
allowed_devices
 *
shape
:0
*
	container 
i
-dense/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense/kernel*
_output_shapes
: 
?
dense/kernel/AssignAssignVariableOpdense/kernel'dense/kernel/Initializer/random_uniform*
dtype0*
validate_shape( 
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
dtype0*
_output_shapes

:0

?
dense/bias/Initializer/zerosConst*
_output_shapes
:
*
_class
loc:@dense/bias*
valueB
*    *
dtype0
?

dense/biasVarHandleOp*
allowed_devices
 *
dtype0*
_class
loc:@dense/bias*
_output_shapes
: *
shared_name
dense/bias*
	container *
shape:

e
+dense/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOp
dense/bias*
_output_shapes
: 
r
dense/bias/AssignAssignVariableOp
dense/biasdense/bias/Initializer/zeros*
dtype0*
validate_shape( 
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:
*
dtype0
h
dense/MatMul/ReadVariableOpReadVariableOpdense/kernel*
dtype0*
_output_shapes

:0

?
dense/MatMulMatMulstream_6/flatten/Reshapedense/MatMul/ReadVariableOp*
transpose_a( *
_output_shapes

:
*
transpose_b( *
T0
c
dense/BiasAdd/ReadVariableOpReadVariableOp
dense/bias*
dtype0*
_output_shapes
:

?
dense/BiasAddBiasAdddense/MatMuldense/BiasAdd/ReadVariableOp*
_output_shapes

:
*
data_formatNHWC*
T0
J

dense/ReluReludense/BiasAdd*
T0*
_output_shapes

:

?
/dense_1/kernel/Initializer/random_uniform/shapeConst*
valueB"
      *!
_class
loc:@dense_1/kernel*
_output_shapes
:*
dtype0
?
-dense_1/kernel/Initializer/random_uniform/minConst*
dtype0*!
_class
loc:@dense_1/kernel*
valueB
 *?5?*
_output_shapes
: 
?
-dense_1/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *!
_class
loc:@dense_1/kernel*
dtype0*
valueB
 *?5?
?
7dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_1/kernel/Initializer/random_uniform/shape*
dtype0*!
_class
loc:@dense_1/kernel*
T0*
_output_shapes

:
*
seed2 *

seed 
?
-dense_1/kernel/Initializer/random_uniform/subSub-dense_1/kernel/Initializer/random_uniform/max-dense_1/kernel/Initializer/random_uniform/min*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes
: 
?
-dense_1/kernel/Initializer/random_uniform/mulMul7dense_1/kernel/Initializer/random_uniform/RandomUniform-dense_1/kernel/Initializer/random_uniform/sub*
T0*
_output_shapes

:
*!
_class
loc:@dense_1/kernel
?
)dense_1/kernel/Initializer/random_uniformAddV2-dense_1/kernel/Initializer/random_uniform/mul-dense_1/kernel/Initializer/random_uniform/min*
_output_shapes

:
*
T0*!
_class
loc:@dense_1/kernel
?
dense_1/kernelVarHandleOp*
	container *
shape
:
*
_output_shapes
: *!
_class
loc:@dense_1/kernel*
dtype0*
allowed_devices
 *
shared_namedense_1/kernel
m
/dense_1/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_1/kernel*
_output_shapes
: 
?
dense_1/kernel/AssignAssignVariableOpdense_1/kernel)dense_1/kernel/Initializer/random_uniform*
dtype0*
validate_shape( 
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
dtype0*
_output_shapes

:

?
dense_1/bias/Initializer/zerosConst*
_output_shapes
:*
valueB*    *
_class
loc:@dense_1/bias*
dtype0
?
dense_1/biasVarHandleOp*
_class
loc:@dense_1/bias*
allowed_devices
 *
shape:*
shared_namedense_1/bias*
_output_shapes
: *
	container *
dtype0
i
-dense_1/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_1/bias*
_output_shapes
: 
x
dense_1/bias/AssignAssignVariableOpdense_1/biasdense_1/bias/Initializer/zeros*
dtype0*
validate_shape( 
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:*
dtype0
l
dense_1/MatMul/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

:
*
dtype0
?
dense_1/MatMulMatMul
dense/Reludense_1/MatMul/ReadVariableOp*
transpose_b( *
T0*
_output_shapes

:*
transpose_a( 
g
dense_1/BiasAdd/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:*
dtype0
?
dense_1/BiasAddBiasAdddense_1/MatMuldense_1/BiasAdd/ReadVariableOp*
_output_shapes

:*
T0*
data_formatNHWC
T
dense_1/SoftmaxSoftmaxdense_1/BiasAdd*
_output_shapes

:*
T0ƪ
?
?
#batch_normalization_2_cond_true_2898
*readvariableop_batch_normalization_2_gamma:9
+readvariableop_1_batch_normalization_2_beta:O
Afusedbatchnormv3_readvariableop_batch_normalization_2_moving_mean:U
Gfusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance:8
4fusedbatchnormv3_stream_2_depthwise_conv2d_1_biasadd
fusedbatchnormv3
fusedbatchnormv3_0
fusedbatchnormv3_1
optionalfromvalue
optionalfromvalue_1
optionalfromvalue_2
optionalfromvalue_3
optionalfromvalue_4
optionalfromvalue_5
optionalfromvalue_6?u
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_2_gamma*
_output_shapes
:*
dtype0x
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_2_beta*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_2_moving_mean*
dtype0*
_output_shapes
:?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV34fusedbatchnormv3_stream_2_depthwise_conv2d_1_biasaddReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
data_formatNHWC*
U0*
epsilon%??'7*B
_output_shapes0
.:9:::::*
T0*
is_training(*
exponential_avg_factor%  ???
OptionalFromValueOptionalFromValueReadVariableOp:value:0*
_output_shapes
: *
Toutput_types
2:????
OptionalFromValue_1OptionalFromValueReadVariableOp_1:value:0*
_output_shapes
: *
Toutput_types
2:????
OptionalFromValue_2OptionalFromValue'FusedBatchNormV3/ReadVariableOp:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_3OptionalFromValue)FusedBatchNormV3/ReadVariableOp_1:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_4OptionalFromValue"FusedBatchNormV3:reserve_space_1:0*
_output_shapes
: *
Toutput_types
2:????
OptionalFromValue_5OptionalFromValue"FusedBatchNormV3:reserve_space_2:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_6OptionalFromValue"FusedBatchNormV3:reserve_space_3:0*
Toutput_types
2*
_output_shapes
: :???"3
fusedbatchnormv3_0FusedBatchNormV3:batch_mean:0"7
fusedbatchnormv3_1!FusedBatchNormV3:batch_variance:0"1
optionalfromvalueOptionalFromValue:optional:0"(
fusedbatchnormv3FusedBatchNormV3:y:0"5
optionalfromvalue_1OptionalFromValue_1:optional:0"5
optionalfromvalue_2OptionalFromValue_2:optional:0"5
optionalfromvalue_3OptionalFromValue_3:optional:0"5
optionalfromvalue_4OptionalFromValue_4:optional:0"5
optionalfromvalue_5OptionalFromValue_5:optional:0"5
optionalfromvalue_6OptionalFromValue_6:optional:0*-
_input_shapes
: : : : :9:,(
&
_output_shapes
:9
?
2
%batch_normalization_4_cond_1_true_608	
constJ
ConstConst*
dtype0*
_output_shapes
: *
valueB
 *???="
constConst:output:0*
_input_shapes 
?
?
#batch_normalization_4_cond_true_5408
*readvariableop_batch_normalization_4_gamma:9
+readvariableop_1_batch_normalization_4_beta:O
Afusedbatchnormv3_readvariableop_batch_normalization_4_moving_mean:U
Gfusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance:8
4fusedbatchnormv3_stream_5_depthwise_conv2d_3_biasadd
fusedbatchnormv3
fusedbatchnormv3_0
fusedbatchnormv3_1
optionalfromvalue
optionalfromvalue_1
optionalfromvalue_2
optionalfromvalue_3
optionalfromvalue_4
optionalfromvalue_5
optionalfromvalue_6?u
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_4_gamma*
dtype0*
_output_shapes
:x
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_4_beta*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_4_moving_mean*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV34fusedbatchnormv3_stream_5_depthwise_conv2d_3_biasaddReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*B
_output_shapes0
.::::::*
epsilon%??'7*
data_formatNHWC*
is_training(*
T0*
exponential_avg_factor%  ??*
U0?
OptionalFromValueOptionalFromValueReadVariableOp:value:0*
_output_shapes
: *
Toutput_types
2:????
OptionalFromValue_1OptionalFromValueReadVariableOp_1:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_2OptionalFromValue'FusedBatchNormV3/ReadVariableOp:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_3OptionalFromValue)FusedBatchNormV3/ReadVariableOp_1:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_4OptionalFromValue"FusedBatchNormV3:reserve_space_1:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_5OptionalFromValue"FusedBatchNormV3:reserve_space_2:0*
_output_shapes
: *
Toutput_types
2:????
OptionalFromValue_6OptionalFromValue"FusedBatchNormV3:reserve_space_3:0*
Toutput_types
2*
_output_shapes
: :???"5
optionalfromvalue_4OptionalFromValue_4:optional:0"5
optionalfromvalue_5OptionalFromValue_5:optional:0"5
optionalfromvalue_6OptionalFromValue_6:optional:0"3
fusedbatchnormv3_0FusedBatchNormV3:batch_mean:0"7
fusedbatchnormv3_1!FusedBatchNormV3:batch_variance:0"1
optionalfromvalueOptionalFromValue:optional:0"(
fusedbatchnormv3FusedBatchNormV3:y:0"5
optionalfromvalue_1OptionalFromValue_1:optional:0"5
optionalfromvalue_2OptionalFromValue_2:optional:0"5
optionalfromvalue_3OptionalFromValue_3:optional:0*-
_input_shapes
: : : : ::,(
&
_output_shapes
:
?
?
$batch_normalization_3_cond_false_4168
*readvariableop_batch_normalization_3_gamma:9
+readvariableop_1_batch_normalization_3_beta:O
Afusedbatchnormv3_readvariableop_batch_normalization_3_moving_mean:U
Gfusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance:8
4fusedbatchnormv3_stream_4_depthwise_conv2d_2_biasadd
fusedbatchnormv3
fusedbatchnormv3_0
fusedbatchnormv3_1
optionalfromvalue
optionalfromvalue_1
optionalfromvalue_2
optionalfromvalue_3
optionalfromvalue_4
optionalfromvalue_5
optionalfromvalue_6?u
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_3_gamma*
dtype0*
_output_shapes
:x
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_3_beta*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_3_moving_mean*
dtype0*
_output_shapes
:?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance*
dtype0*
_output_shapes
:?
FusedBatchNormV3FusedBatchNormV34fusedbatchnormv3_stream_4_depthwise_conv2d_2_biasaddReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
is_training( *
epsilon%??'7*B
_output_shapes0
.::::::*
exponential_avg_factor%  ??*
data_formatNHWC*
U0*
T0?
OptionalFromValueOptionalFromValueReadVariableOp:value:0*
_output_shapes
: *
Toutput_types
2:????
OptionalFromValue_1OptionalFromValueReadVariableOp_1:value:0*
_output_shapes
: *
Toutput_types
2:????
OptionalFromValue_2OptionalFromValue'FusedBatchNormV3/ReadVariableOp:value:0*
_output_shapes
: *
Toutput_types
2:????
OptionalFromValue_3OptionalFromValue)FusedBatchNormV3/ReadVariableOp_1:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_4OptionalFromValue"FusedBatchNormV3:reserve_space_1:0*
_output_shapes
: *
Toutput_types
2:????
OptionalFromValue_5OptionalFromValue"FusedBatchNormV3:reserve_space_2:0*
_output_shapes
: *
Toutput_types
2:????
OptionalFromValue_6OptionalFromValue"FusedBatchNormV3:reserve_space_3:0*
_output_shapes
: *
Toutput_types
2:???"1
optionalfromvalueOptionalFromValue:optional:0"(
fusedbatchnormv3FusedBatchNormV3:y:0"5
optionalfromvalue_1OptionalFromValue_1:optional:0"5
optionalfromvalue_2OptionalFromValue_2:optional:0"5
optionalfromvalue_3OptionalFromValue_3:optional:0"5
optionalfromvalue_4OptionalFromValue_4:optional:0"5
optionalfromvalue_5OptionalFromValue_5:optional:0"5
optionalfromvalue_6OptionalFromValue_6:optional:0"3
fusedbatchnormv3_0FusedBatchNormV3:batch_mean:0"7
fusedbatchnormv3_1!FusedBatchNormV3:batch_variance:0*-
_input_shapes
: : : : ::,(
&
_output_shapes
:
?
3
&batch_normalization_1_cond_1_false_233	
constJ
ConstConst*
dtype0*
valueB
 *  ??*
_output_shapes
: "
constConst:output:0*
_input_shapes 
?
2
%batch_normalization_2_cond_1_true_357	
constJ
ConstConst*
_output_shapes
: *
valueB
 *???=*
dtype0"
constConst:output:0*
_input_shapes 
?
3
&batch_normalization_3_cond_1_false_484	
constJ
ConstConst*
dtype0*
_output_shapes
: *
valueB
 *  ??"
constConst:output:0*
_input_shapes 
?
?
 batch_normalization_cond_true_396
(readvariableop_batch_normalization_gamma:7
)readvariableop_1_batch_normalization_beta:M
?fusedbatchnormv3_readvariableop_batch_normalization_moving_mean:S
Efusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance:*
&fusedbatchnormv3_stream_conv2d_biasadd
fusedbatchnormv3
fusedbatchnormv3_0
fusedbatchnormv3_1
optionalfromvalue
optionalfromvalue_1
optionalfromvalue_2
optionalfromvalue_3
optionalfromvalue_4
optionalfromvalue_5
optionalfromvalue_6?s
ReadVariableOpReadVariableOp(readvariableop_batch_normalization_gamma*
_output_shapes
:*
dtype0v
ReadVariableOp_1ReadVariableOp)readvariableop_1_batch_normalization_beta*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp?fusedbatchnormv3_readvariableop_batch_normalization_moving_mean*
dtype0*
_output_shapes
:?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpEfusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3&fusedbatchnormv3_stream_conv2d_biasaddReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
is_training(*
data_formatNHWC*
T0*
epsilon%??'7*
exponential_avg_factor%  ??*B
_output_shapes0
.:{:::::*
U0?
OptionalFromValueOptionalFromValueReadVariableOp:value:0*
_output_shapes
: *
Toutput_types
2:????
OptionalFromValue_1OptionalFromValueReadVariableOp_1:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_2OptionalFromValue'FusedBatchNormV3/ReadVariableOp:value:0*
_output_shapes
: *
Toutput_types
2:????
OptionalFromValue_3OptionalFromValue)FusedBatchNormV3/ReadVariableOp_1:value:0*
_output_shapes
: *
Toutput_types
2:????
OptionalFromValue_4OptionalFromValue"FusedBatchNormV3:reserve_space_1:0*
_output_shapes
: *
Toutput_types
2:????
OptionalFromValue_5OptionalFromValue"FusedBatchNormV3:reserve_space_2:0*
_output_shapes
: *
Toutput_types
2:????
OptionalFromValue_6OptionalFromValue"FusedBatchNormV3:reserve_space_3:0*
_output_shapes
: *
Toutput_types
2:???"(
fusedbatchnormv3FusedBatchNormV3:y:0"5
optionalfromvalue_1OptionalFromValue_1:optional:0"5
optionalfromvalue_2OptionalFromValue_2:optional:0"5
optionalfromvalue_3OptionalFromValue_3:optional:0"5
optionalfromvalue_4OptionalFromValue_4:optional:0"5
optionalfromvalue_5OptionalFromValue_5:optional:0"5
optionalfromvalue_6OptionalFromValue_6:optional:0"3
fusedbatchnormv3_0FusedBatchNormV3:batch_mean:0"7
fusedbatchnormv3_1!FusedBatchNormV3:batch_variance:0"1
optionalfromvalueOptionalFromValue:optional:0*-
_input_shapes
: : : : :{:,(
&
_output_shapes
:{
?
?
$batch_normalization_2_cond_false_2908
*readvariableop_batch_normalization_2_gamma:9
+readvariableop_1_batch_normalization_2_beta:O
Afusedbatchnormv3_readvariableop_batch_normalization_2_moving_mean:U
Gfusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance:8
4fusedbatchnormv3_stream_2_depthwise_conv2d_1_biasadd
fusedbatchnormv3
fusedbatchnormv3_0
fusedbatchnormv3_1
optionalfromvalue
optionalfromvalue_1
optionalfromvalue_2
optionalfromvalue_3
optionalfromvalue_4
optionalfromvalue_5
optionalfromvalue_6?u
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_2_gamma*
dtype0*
_output_shapes
:x
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_2_beta*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_2_moving_mean*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV34fusedbatchnormv3_stream_2_depthwise_conv2d_1_biasaddReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
exponential_avg_factor%  ??*
T0*
data_formatNHWC*
U0*
is_training( *B
_output_shapes0
.:9:::::*
epsilon%??'7?
OptionalFromValueOptionalFromValueReadVariableOp:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_1OptionalFromValueReadVariableOp_1:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_2OptionalFromValue'FusedBatchNormV3/ReadVariableOp:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_3OptionalFromValue)FusedBatchNormV3/ReadVariableOp_1:value:0*
_output_shapes
: *
Toutput_types
2:????
OptionalFromValue_4OptionalFromValue"FusedBatchNormV3:reserve_space_1:0*
_output_shapes
: *
Toutput_types
2:????
OptionalFromValue_5OptionalFromValue"FusedBatchNormV3:reserve_space_2:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_6OptionalFromValue"FusedBatchNormV3:reserve_space_3:0*
Toutput_types
2*
_output_shapes
: :???"5
optionalfromvalue_6OptionalFromValue_6:optional:0"3
fusedbatchnormv3_0FusedBatchNormV3:batch_mean:0"7
fusedbatchnormv3_1!FusedBatchNormV3:batch_variance:0"1
optionalfromvalueOptionalFromValue:optional:0"(
fusedbatchnormv3FusedBatchNormV3:y:0"5
optionalfromvalue_1OptionalFromValue_1:optional:0"5
optionalfromvalue_2OptionalFromValue_2:optional:0"5
optionalfromvalue_3OptionalFromValue_3:optional:0"5
optionalfromvalue_4OptionalFromValue_4:optional:0"5
optionalfromvalue_5OptionalFromValue_5:optional:0*-
_input_shapes
: : : : :9:,(
&
_output_shapes
:9
?
?
#batch_normalization_1_cond_true_1648
*readvariableop_batch_normalization_1_gamma:9
+readvariableop_1_batch_normalization_1_beta:O
Afusedbatchnormv3_readvariableop_batch_normalization_1_moving_mean:U
Gfusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance:6
2fusedbatchnormv3_stream_1_depthwise_conv2d_biasadd
fusedbatchnormv3
fusedbatchnormv3_0
fusedbatchnormv3_1
optionalfromvalue
optionalfromvalue_1
optionalfromvalue_2
optionalfromvalue_3
optionalfromvalue_4
optionalfromvalue_5
optionalfromvalue_6?u
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_1_gamma*
dtype0*
_output_shapes
:x
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_1_beta*
dtype0*
_output_shapes
:?
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_1_moving_mean*
dtype0*
_output_shapes
:?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance*
dtype0*
_output_shapes
:?
FusedBatchNormV3FusedBatchNormV32fusedbatchnormv3_stream_1_depthwise_conv2d_biasaddReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
data_formatNHWC*
T0*
epsilon%??'7*
is_training(*
exponential_avg_factor%  ??*B
_output_shapes0
.:<:::::*
U0?
OptionalFromValueOptionalFromValueReadVariableOp:value:0*
_output_shapes
: *
Toutput_types
2:????
OptionalFromValue_1OptionalFromValueReadVariableOp_1:value:0*
_output_shapes
: *
Toutput_types
2:????
OptionalFromValue_2OptionalFromValue'FusedBatchNormV3/ReadVariableOp:value:0*
_output_shapes
: *
Toutput_types
2:????
OptionalFromValue_3OptionalFromValue)FusedBatchNormV3/ReadVariableOp_1:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_4OptionalFromValue"FusedBatchNormV3:reserve_space_1:0*
_output_shapes
: *
Toutput_types
2:????
OptionalFromValue_5OptionalFromValue"FusedBatchNormV3:reserve_space_2:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_6OptionalFromValue"FusedBatchNormV3:reserve_space_3:0*
_output_shapes
: *
Toutput_types
2:???"7
fusedbatchnormv3_1!FusedBatchNormV3:batch_variance:0"1
optionalfromvalueOptionalFromValue:optional:0"(
fusedbatchnormv3FusedBatchNormV3:y:0"5
optionalfromvalue_1OptionalFromValue_1:optional:0"5
optionalfromvalue_2OptionalFromValue_2:optional:0"5
optionalfromvalue_3OptionalFromValue_3:optional:0"5
optionalfromvalue_4OptionalFromValue_4:optional:0"5
optionalfromvalue_5OptionalFromValue_5:optional:0"5
optionalfromvalue_6OptionalFromValue_6:optional:0"3
fusedbatchnormv3_0FusedBatchNormV3:batch_mean:0*-
_input_shapes
: : : : :<:,(
&
_output_shapes
:<
?
?
dropout_cond_true_629!
dropout_mul_activation_4_relu
dropout_mul_1
optionalfromvalue
optionalfromvalue_1
optionalfromvalue_2
optionalfromvalue_3
optionalfromvalue_4
optionalfromvalue_5
optionalfromvalue_6?R
dropout/ConstConst*
dtype0*
valueB
 *   @*
_output_shapes
: z
dropout/MulMuldropout_mul_activation_4_reludropout/Const:output:0*
T0*&
_output_shapes
:f
dropout/ShapeConst*
_output_shapes
:*%
valueB"            *
dtype0?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
seed2 *&
_output_shapes
:*
dtype0*

seed *
T0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*&
_output_shapes
:~
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*
Truncate( *&
_output_shapes
:*

SrcT0
h
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*&
_output_shapes
:*
T0?
OptionalFromValueOptionalFromValuedropout/Const:output:0*
_output_shapes
: *
Toutput_types
2:????
OptionalFromValue_1OptionalFromValuedropout/Mul:z:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_2OptionalFromValuedropout/Shape:output:0*
_output_shapes
: *
Toutput_types
2:????
OptionalFromValue_3OptionalFromValue-dropout/random_uniform/RandomUniform:output:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_4OptionalFromValuedropout/GreaterEqual/y:output:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_5OptionalFromValuedropout/GreaterEqual:z:0*
_output_shapes
: *
Toutput_types
2
:????
OptionalFromValue_6OptionalFromValuedropout/Cast:y:0*
_output_shapes
: *
Toutput_types
2:???"5
optionalfromvalue_2OptionalFromValue_2:optional:0"5
optionalfromvalue_3OptionalFromValue_3:optional:0"5
optionalfromvalue_4OptionalFromValue_4:optional:0"5
optionalfromvalue_5OptionalFromValue_5:optional:0"5
optionalfromvalue_6OptionalFromValue_6:optional:0""
dropout_mul_1dropout/Mul_1:z:0"1
optionalfromvalueOptionalFromValue:optional:0"5
optionalfromvalue_1OptionalFromValue_1:optional:0*%
_input_shapes
::, (
&
_output_shapes
:
?
3
&batch_normalization_2_cond_1_false_358	
constJ
ConstConst*
valueB
 *  ??*
_output_shapes
: *
dtype0"
constConst:output:0*
_input_shapes 
?
?
!batch_normalization_cond_false_406
(readvariableop_batch_normalization_gamma:7
)readvariableop_1_batch_normalization_beta:M
?fusedbatchnormv3_readvariableop_batch_normalization_moving_mean:S
Efusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance:*
&fusedbatchnormv3_stream_conv2d_biasadd
fusedbatchnormv3
fusedbatchnormv3_0
fusedbatchnormv3_1
optionalfromvalue
optionalfromvalue_1
optionalfromvalue_2
optionalfromvalue_3
optionalfromvalue_4
optionalfromvalue_5
optionalfromvalue_6?s
ReadVariableOpReadVariableOp(readvariableop_batch_normalization_gamma*
dtype0*
_output_shapes
:v
ReadVariableOp_1ReadVariableOp)readvariableop_1_batch_normalization_beta*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp?fusedbatchnormv3_readvariableop_batch_normalization_moving_mean*
dtype0*
_output_shapes
:?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpEfusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3&fusedbatchnormv3_stream_conv2d_biasaddReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
exponential_avg_factor%  ??*
data_formatNHWC*
T0*B
_output_shapes0
.:{:::::*
U0*
epsilon%??'7*
is_training( ?
OptionalFromValueOptionalFromValueReadVariableOp:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_1OptionalFromValueReadVariableOp_1:value:0*
_output_shapes
: *
Toutput_types
2:????
OptionalFromValue_2OptionalFromValue'FusedBatchNormV3/ReadVariableOp:value:0*
_output_shapes
: *
Toutput_types
2:????
OptionalFromValue_3OptionalFromValue)FusedBatchNormV3/ReadVariableOp_1:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_4OptionalFromValue"FusedBatchNormV3:reserve_space_1:0*
_output_shapes
: *
Toutput_types
2:????
OptionalFromValue_5OptionalFromValue"FusedBatchNormV3:reserve_space_2:0*
_output_shapes
: *
Toutput_types
2:????
OptionalFromValue_6OptionalFromValue"FusedBatchNormV3:reserve_space_3:0*
Toutput_types
2*
_output_shapes
: :???"1
optionalfromvalueOptionalFromValue:optional:0"(
fusedbatchnormv3FusedBatchNormV3:y:0"5
optionalfromvalue_1OptionalFromValue_1:optional:0"5
optionalfromvalue_2OptionalFromValue_2:optional:0"5
optionalfromvalue_3OptionalFromValue_3:optional:0"5
optionalfromvalue_4OptionalFromValue_4:optional:0"5
optionalfromvalue_5OptionalFromValue_5:optional:0"5
optionalfromvalue_6OptionalFromValue_6:optional:0"3
fusedbatchnormv3_0FusedBatchNormV3:batch_mean:0"7
fusedbatchnormv3_1!FusedBatchNormV3:batch_variance:0*-
_input_shapes
: : : : :{:,(
&
_output_shapes
:{
?
?
#batch_normalization_3_cond_true_4158
*readvariableop_batch_normalization_3_gamma:9
+readvariableop_1_batch_normalization_3_beta:O
Afusedbatchnormv3_readvariableop_batch_normalization_3_moving_mean:U
Gfusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance:8
4fusedbatchnormv3_stream_4_depthwise_conv2d_2_biasadd
fusedbatchnormv3
fusedbatchnormv3_0
fusedbatchnormv3_1
optionalfromvalue
optionalfromvalue_1
optionalfromvalue_2
optionalfromvalue_3
optionalfromvalue_4
optionalfromvalue_5
optionalfromvalue_6?u
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_3_gamma*
dtype0*
_output_shapes
:x
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_3_beta*
dtype0*
_output_shapes
:?
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_3_moving_mean*
dtype0*
_output_shapes
:?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance*
dtype0*
_output_shapes
:?
FusedBatchNormV3FusedBatchNormV34fusedbatchnormv3_stream_4_depthwise_conv2d_2_biasaddReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
is_training(*
exponential_avg_factor%  ??*B
_output_shapes0
.::::::*
data_formatNHWC*
U0*
T0*
epsilon%??'7?
OptionalFromValueOptionalFromValueReadVariableOp:value:0*
_output_shapes
: *
Toutput_types
2:????
OptionalFromValue_1OptionalFromValueReadVariableOp_1:value:0*
_output_shapes
: *
Toutput_types
2:????
OptionalFromValue_2OptionalFromValue'FusedBatchNormV3/ReadVariableOp:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_3OptionalFromValue)FusedBatchNormV3/ReadVariableOp_1:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_4OptionalFromValue"FusedBatchNormV3:reserve_space_1:0*
_output_shapes
: *
Toutput_types
2:????
OptionalFromValue_5OptionalFromValue"FusedBatchNormV3:reserve_space_2:0*
_output_shapes
: *
Toutput_types
2:????
OptionalFromValue_6OptionalFromValue"FusedBatchNormV3:reserve_space_3:0*
Toutput_types
2*
_output_shapes
: :???"3
fusedbatchnormv3_0FusedBatchNormV3:batch_mean:0"7
fusedbatchnormv3_1!FusedBatchNormV3:batch_variance:0"1
optionalfromvalueOptionalFromValue:optional:0"(
fusedbatchnormv3FusedBatchNormV3:y:0"5
optionalfromvalue_1OptionalFromValue_1:optional:0"5
optionalfromvalue_2OptionalFromValue_2:optional:0"5
optionalfromvalue_3OptionalFromValue_3:optional:0"5
optionalfromvalue_4OptionalFromValue_4:optional:0"5
optionalfromvalue_5OptionalFromValue_5:optional:0"5
optionalfromvalue_6OptionalFromValue_6:optional:0*-
_input_shapes
: : : : ::,(
&
_output_shapes
:
?
0
#batch_normalization_cond_1_true_107	
constJ
ConstConst*
_output_shapes
: *
valueB
 *???=*
dtype0"
constConst:output:0*
_input_shapes 
?
1
$batch_normalization_cond_1_false_108	
constJ
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??"
constConst:output:0*
_input_shapes 
?
2
%batch_normalization_1_cond_1_true_232	
constJ
ConstConst*
dtype0*
valueB
 *???=*
_output_shapes
: "
constConst:output:0*
_input_shapes 
?
?
$batch_normalization_4_cond_false_5418
*readvariableop_batch_normalization_4_gamma:9
+readvariableop_1_batch_normalization_4_beta:O
Afusedbatchnormv3_readvariableop_batch_normalization_4_moving_mean:U
Gfusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance:8
4fusedbatchnormv3_stream_5_depthwise_conv2d_3_biasadd
fusedbatchnormv3
fusedbatchnormv3_0
fusedbatchnormv3_1
optionalfromvalue
optionalfromvalue_1
optionalfromvalue_2
optionalfromvalue_3
optionalfromvalue_4
optionalfromvalue_5
optionalfromvalue_6?u
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_4_gamma*
_output_shapes
:*
dtype0x
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_4_beta*
dtype0*
_output_shapes
:?
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_4_moving_mean*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance*
dtype0*
_output_shapes
:?
FusedBatchNormV3FusedBatchNormV34fusedbatchnormv3_stream_5_depthwise_conv2d_3_biasaddReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
is_training( *
epsilon%??'7*
exponential_avg_factor%  ??*
U0*
T0*B
_output_shapes0
.::::::*
data_formatNHWC?
OptionalFromValueOptionalFromValueReadVariableOp:value:0*
_output_shapes
: *
Toutput_types
2:????
OptionalFromValue_1OptionalFromValueReadVariableOp_1:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_2OptionalFromValue'FusedBatchNormV3/ReadVariableOp:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_3OptionalFromValue)FusedBatchNormV3/ReadVariableOp_1:value:0*
_output_shapes
: *
Toutput_types
2:????
OptionalFromValue_4OptionalFromValue"FusedBatchNormV3:reserve_space_1:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_5OptionalFromValue"FusedBatchNormV3:reserve_space_2:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_6OptionalFromValue"FusedBatchNormV3:reserve_space_3:0*
_output_shapes
: *
Toutput_types
2:???"3
fusedbatchnormv3_0FusedBatchNormV3:batch_mean:0"7
fusedbatchnormv3_1!FusedBatchNormV3:batch_variance:0"1
optionalfromvalueOptionalFromValue:optional:0"(
fusedbatchnormv3FusedBatchNormV3:y:0"5
optionalfromvalue_1OptionalFromValue_1:optional:0"5
optionalfromvalue_2OptionalFromValue_2:optional:0"5
optionalfromvalue_3OptionalFromValue_3:optional:0"5
optionalfromvalue_4OptionalFromValue_4:optional:0"5
optionalfromvalue_5OptionalFromValue_5:optional:0"5
optionalfromvalue_6OptionalFromValue_6:optional:0*-
_input_shapes
: : : : ::,(
&
_output_shapes
:
?
?
dropout_cond_false_630
identity_activation_4_relu
identity
optionalnone
optionalnone_1
optionalnone_2
optionalnone_3
optionalnone_4
optionalnone_5
optionalnone_6a
IdentityIdentityidentity_activation_4_relu*&
_output_shapes
:*
T04
OptionalNoneOptionalNone*
_output_shapes
: 6
OptionalNone_1OptionalNone*
_output_shapes
: 6
OptionalNone_2OptionalNone*
_output_shapes
: 6
OptionalNone_3OptionalNone*
_output_shapes
: 6
OptionalNone_4OptionalNone*
_output_shapes
: 6
OptionalNone_5OptionalNone*
_output_shapes
: 6
OptionalNone_6OptionalNone*
_output_shapes
: "+
optionalnone_6OptionalNone_6:optional:0"
identityIdentity:output:0"+
optionalnone_1OptionalNone_1:optional:0"+
optionalnone_2OptionalNone_2:optional:0"+
optionalnone_3OptionalNone_3:optional:0"+
optionalnone_4OptionalNone_4:optional:0"+
optionalnone_5OptionalNone_5:optional:0"'
optionalnoneOptionalNone:optional:0*%
_input_shapes
::, (
&
_output_shapes
:
?
?
$batch_normalization_1_cond_false_1658
*readvariableop_batch_normalization_1_gamma:9
+readvariableop_1_batch_normalization_1_beta:O
Afusedbatchnormv3_readvariableop_batch_normalization_1_moving_mean:U
Gfusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance:6
2fusedbatchnormv3_stream_1_depthwise_conv2d_biasadd
fusedbatchnormv3
fusedbatchnormv3_0
fusedbatchnormv3_1
optionalfromvalue
optionalfromvalue_1
optionalfromvalue_2
optionalfromvalue_3
optionalfromvalue_4
optionalfromvalue_5
optionalfromvalue_6?u
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_1_gamma*
_output_shapes
:*
dtype0x
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_1_beta*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_1_moving_mean*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance*
dtype0*
_output_shapes
:?
FusedBatchNormV3FusedBatchNormV32fusedbatchnormv3_stream_1_depthwise_conv2d_biasaddReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
exponential_avg_factor%  ??*
data_formatNHWC*
U0*
is_training( *B
_output_shapes0
.:<:::::*
epsilon%??'7?
OptionalFromValueOptionalFromValueReadVariableOp:value:0*
_output_shapes
: *
Toutput_types
2:????
OptionalFromValue_1OptionalFromValueReadVariableOp_1:value:0*
_output_shapes
: *
Toutput_types
2:????
OptionalFromValue_2OptionalFromValue'FusedBatchNormV3/ReadVariableOp:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_3OptionalFromValue)FusedBatchNormV3/ReadVariableOp_1:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_4OptionalFromValue"FusedBatchNormV3:reserve_space_1:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_5OptionalFromValue"FusedBatchNormV3:reserve_space_2:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_6OptionalFromValue"FusedBatchNormV3:reserve_space_3:0*
Toutput_types
2*
_output_shapes
: :???"5
optionalfromvalue_3OptionalFromValue_3:optional:0"5
optionalfromvalue_4OptionalFromValue_4:optional:0"5
optionalfromvalue_5OptionalFromValue_5:optional:0"5
optionalfromvalue_6OptionalFromValue_6:optional:0"3
fusedbatchnormv3_0FusedBatchNormV3:batch_mean:0"7
fusedbatchnormv3_1!FusedBatchNormV3:batch_variance:0"1
optionalfromvalueOptionalFromValue:optional:0"(
fusedbatchnormv3FusedBatchNormV3:y:0"5
optionalfromvalue_1OptionalFromValue_1:optional:0"5
optionalfromvalue_2OptionalFromValue_2:optional:0*-
_input_shapes
: : : : :<:,(
&
_output_shapes
:<
?
2
%batch_normalization_3_cond_1_true_483	
constJ
ConstConst*
dtype0*
_output_shapes
: *
valueB
 *???="
constConst:output:0*
_input_shapes 
?
3
&batch_normalization_4_cond_1_false_609	
constJ
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??"
constConst:output:0*
_input_shapes "?"?
trainable_variables??
|
stream/kernel:0stream/kernel/Assign#stream/kernel/Read/ReadVariableOp:0(2*stream/kernel/Initializer/random_uniform:08
k
stream/bias:0stream/bias/Assign!stream/bias/Read/ReadVariableOp:0(2stream/bias/Initializer/zeros:08
?
batch_normalization/gamma:0 batch_normalization/gamma/Assign/batch_normalization/gamma/Read/ReadVariableOp:0(2,batch_normalization/gamma/Initializer/ones:08
?
batch_normalization/beta:0batch_normalization/beta/Assign.batch_normalization/beta/Read/ReadVariableOp:0(2,batch_normalization/beta/Initializer/zeros:08
?
stream_1/depthwise_kernel:0 stream_1/depthwise_kernel/Assign/stream_1/depthwise_kernel/Read/ReadVariableOp:0(26stream_1/depthwise_kernel/Initializer/random_uniform:08
s
stream_1/bias:0stream_1/bias/Assign#stream_1/bias/Read/ReadVariableOp:0(2!stream_1/bias/Initializer/zeros:08
?
batch_normalization_1/gamma:0"batch_normalization_1/gamma/Assign1batch_normalization_1/gamma/Read/ReadVariableOp:0(2.batch_normalization_1/gamma/Initializer/ones:08
?
batch_normalization_1/beta:0!batch_normalization_1/beta/Assign0batch_normalization_1/beta/Read/ReadVariableOp:0(2.batch_normalization_1/beta/Initializer/zeros:08
?
stream_2/depthwise_kernel:0 stream_2/depthwise_kernel/Assign/stream_2/depthwise_kernel/Read/ReadVariableOp:0(26stream_2/depthwise_kernel/Initializer/random_uniform:08
s
stream_2/bias:0stream_2/bias/Assign#stream_2/bias/Read/ReadVariableOp:0(2!stream_2/bias/Initializer/zeros:08
?
batch_normalization_2/gamma:0"batch_normalization_2/gamma/Assign1batch_normalization_2/gamma/Read/ReadVariableOp:0(2.batch_normalization_2/gamma/Initializer/ones:08
?
batch_normalization_2/beta:0!batch_normalization_2/beta/Assign0batch_normalization_2/beta/Read/ReadVariableOp:0(2.batch_normalization_2/beta/Initializer/zeros:08
?
stream_4/depthwise_kernel:0 stream_4/depthwise_kernel/Assign/stream_4/depthwise_kernel/Read/ReadVariableOp:0(26stream_4/depthwise_kernel/Initializer/random_uniform:08
s
stream_4/bias:0stream_4/bias/Assign#stream_4/bias/Read/ReadVariableOp:0(2!stream_4/bias/Initializer/zeros:08
?
batch_normalization_3/gamma:0"batch_normalization_3/gamma/Assign1batch_normalization_3/gamma/Read/ReadVariableOp:0(2.batch_normalization_3/gamma/Initializer/ones:08
?
batch_normalization_3/beta:0!batch_normalization_3/beta/Assign0batch_normalization_3/beta/Read/ReadVariableOp:0(2.batch_normalization_3/beta/Initializer/zeros:08
?
stream_5/depthwise_kernel:0 stream_5/depthwise_kernel/Assign/stream_5/depthwise_kernel/Read/ReadVariableOp:0(26stream_5/depthwise_kernel/Initializer/random_uniform:08
s
stream_5/bias:0stream_5/bias/Assign#stream_5/bias/Read/ReadVariableOp:0(2!stream_5/bias/Initializer/zeros:08
?
batch_normalization_4/gamma:0"batch_normalization_4/gamma/Assign1batch_normalization_4/gamma/Read/ReadVariableOp:0(2.batch_normalization_4/gamma/Initializer/ones:08
?
batch_normalization_4/beta:0!batch_normalization_4/beta/Assign0batch_normalization_4/beta/Read/ReadVariableOp:0(2.batch_normalization_4/beta/Initializer/zeros:08
x
dense/kernel:0dense/kernel/Assign"dense/kernel/Read/ReadVariableOp:0(2)dense/kernel/Initializer/random_uniform:08
g
dense/bias:0dense/bias/Assign dense/bias/Read/ReadVariableOp:0(2dense/bias/Initializer/zeros:08
?
dense_1/kernel:0dense_1/kernel/Assign$dense_1/kernel/Read/ReadVariableOp:0(2+dense_1/kernel/Initializer/random_uniform:08
o
dense_1/bias:0dense_1/bias/Assign"dense_1/bias/Read/ReadVariableOp:0(2 dense_1/bias/Initializer/zeros:08"?,
	variables?,?,
|
stream/kernel:0stream/kernel/Assign#stream/kernel/Read/ReadVariableOp:0(2*stream/kernel/Initializer/random_uniform:08
k
stream/bias:0stream/bias/Assign!stream/bias/Read/ReadVariableOp:0(2stream/bias/Initializer/zeros:08
?
batch_normalization/gamma:0 batch_normalization/gamma/Assign/batch_normalization/gamma/Read/ReadVariableOp:0(2,batch_normalization/gamma/Initializer/ones:08
?
batch_normalization/beta:0batch_normalization/beta/Assign.batch_normalization/beta/Read/ReadVariableOp:0(2,batch_normalization/beta/Initializer/zeros:08
?
!batch_normalization/moving_mean:0&batch_normalization/moving_mean/Assign5batch_normalization/moving_mean/Read/ReadVariableOp:0(23batch_normalization/moving_mean/Initializer/zeros:0@H
?
%batch_normalization/moving_variance:0*batch_normalization/moving_variance/Assign9batch_normalization/moving_variance/Read/ReadVariableOp:0(26batch_normalization/moving_variance/Initializer/ones:0@H
?
stream_1/depthwise_kernel:0 stream_1/depthwise_kernel/Assign/stream_1/depthwise_kernel/Read/ReadVariableOp:0(26stream_1/depthwise_kernel/Initializer/random_uniform:08
s
stream_1/bias:0stream_1/bias/Assign#stream_1/bias/Read/ReadVariableOp:0(2!stream_1/bias/Initializer/zeros:08
?
batch_normalization_1/gamma:0"batch_normalization_1/gamma/Assign1batch_normalization_1/gamma/Read/ReadVariableOp:0(2.batch_normalization_1/gamma/Initializer/ones:08
?
batch_normalization_1/beta:0!batch_normalization_1/beta/Assign0batch_normalization_1/beta/Read/ReadVariableOp:0(2.batch_normalization_1/beta/Initializer/zeros:08
?
#batch_normalization_1/moving_mean:0(batch_normalization_1/moving_mean/Assign7batch_normalization_1/moving_mean/Read/ReadVariableOp:0(25batch_normalization_1/moving_mean/Initializer/zeros:0@H
?
'batch_normalization_1/moving_variance:0,batch_normalization_1/moving_variance/Assign;batch_normalization_1/moving_variance/Read/ReadVariableOp:0(28batch_normalization_1/moving_variance/Initializer/ones:0@H
?
stream_2/depthwise_kernel:0 stream_2/depthwise_kernel/Assign/stream_2/depthwise_kernel/Read/ReadVariableOp:0(26stream_2/depthwise_kernel/Initializer/random_uniform:08
s
stream_2/bias:0stream_2/bias/Assign#stream_2/bias/Read/ReadVariableOp:0(2!stream_2/bias/Initializer/zeros:08
?
batch_normalization_2/gamma:0"batch_normalization_2/gamma/Assign1batch_normalization_2/gamma/Read/ReadVariableOp:0(2.batch_normalization_2/gamma/Initializer/ones:08
?
batch_normalization_2/beta:0!batch_normalization_2/beta/Assign0batch_normalization_2/beta/Read/ReadVariableOp:0(2.batch_normalization_2/beta/Initializer/zeros:08
?
#batch_normalization_2/moving_mean:0(batch_normalization_2/moving_mean/Assign7batch_normalization_2/moving_mean/Read/ReadVariableOp:0(25batch_normalization_2/moving_mean/Initializer/zeros:0@H
?
'batch_normalization_2/moving_variance:0,batch_normalization_2/moving_variance/Assign;batch_normalization_2/moving_variance/Read/ReadVariableOp:0(28batch_normalization_2/moving_variance/Initializer/ones:0@H
?
stream_4/depthwise_kernel:0 stream_4/depthwise_kernel/Assign/stream_4/depthwise_kernel/Read/ReadVariableOp:0(26stream_4/depthwise_kernel/Initializer/random_uniform:08
s
stream_4/bias:0stream_4/bias/Assign#stream_4/bias/Read/ReadVariableOp:0(2!stream_4/bias/Initializer/zeros:08
?
batch_normalization_3/gamma:0"batch_normalization_3/gamma/Assign1batch_normalization_3/gamma/Read/ReadVariableOp:0(2.batch_normalization_3/gamma/Initializer/ones:08
?
batch_normalization_3/beta:0!batch_normalization_3/beta/Assign0batch_normalization_3/beta/Read/ReadVariableOp:0(2.batch_normalization_3/beta/Initializer/zeros:08
?
#batch_normalization_3/moving_mean:0(batch_normalization_3/moving_mean/Assign7batch_normalization_3/moving_mean/Read/ReadVariableOp:0(25batch_normalization_3/moving_mean/Initializer/zeros:0@H
?
'batch_normalization_3/moving_variance:0,batch_normalization_3/moving_variance/Assign;batch_normalization_3/moving_variance/Read/ReadVariableOp:0(28batch_normalization_3/moving_variance/Initializer/ones:0@H
?
stream_5/depthwise_kernel:0 stream_5/depthwise_kernel/Assign/stream_5/depthwise_kernel/Read/ReadVariableOp:0(26stream_5/depthwise_kernel/Initializer/random_uniform:08
s
stream_5/bias:0stream_5/bias/Assign#stream_5/bias/Read/ReadVariableOp:0(2!stream_5/bias/Initializer/zeros:08
?
batch_normalization_4/gamma:0"batch_normalization_4/gamma/Assign1batch_normalization_4/gamma/Read/ReadVariableOp:0(2.batch_normalization_4/gamma/Initializer/ones:08
?
batch_normalization_4/beta:0!batch_normalization_4/beta/Assign0batch_normalization_4/beta/Read/ReadVariableOp:0(2.batch_normalization_4/beta/Initializer/zeros:08
?
#batch_normalization_4/moving_mean:0(batch_normalization_4/moving_mean/Assign7batch_normalization_4/moving_mean/Read/ReadVariableOp:0(25batch_normalization_4/moving_mean/Initializer/zeros:0@H
?
'batch_normalization_4/moving_variance:0,batch_normalization_4/moving_variance/Assign;batch_normalization_4/moving_variance/Read/ReadVariableOp:0(28batch_normalization_4/moving_variance/Initializer/ones:0@H
x
dense/kernel:0dense/kernel/Assign"dense/kernel/Read/ReadVariableOp:0(2)dense/kernel/Initializer/random_uniform:08
g
dense/bias:0dense/bias/Assign dense/bias/Read/ReadVariableOp:0(2dense/bias/Initializer/zeros:08
?
dense_1/kernel:0dense_1/kernel/Assign$dense_1/kernel/Read/ReadVariableOp:0(2+dense_1/kernel/Initializer/random_uniform:08
o
dense_1/bias:0dense_1/bias/Assign"dense_1/bias/Read/ReadVariableOp:0(2 dense_1/bias/Initializer/zeros:08:?-