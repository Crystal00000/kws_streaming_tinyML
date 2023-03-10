��
��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
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
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
�
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
�
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
$
DisableCopyOnRead
resource�
�
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%��8"&
exponential_avg_factorfloat%  �?";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
�
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
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
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
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.12.0-dev202302032v1.12.1-88849-g3e5b6f1899f8��
r
stream_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namestream_5/bias
k
!stream_5/bias/Read/ReadVariableOpReadVariableOpstream_5/bias*
_output_shapes
:*
dtype0
�
stream_5/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namestream_5/depthwise_kernel
�
-stream_5/depthwise_kernel/Read/ReadVariableOpReadVariableOpstream_5/depthwise_kernel*&
_output_shapes
:*
dtype0
r
stream_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namestream_4/bias
k
!stream_4/bias/Read/ReadVariableOpReadVariableOpstream_4/bias*
_output_shapes
:*
dtype0
�
stream_4/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namestream_4/depthwise_kernel
�
-stream_4/depthwise_kernel/Read/ReadVariableOpReadVariableOpstream_4/depthwise_kernel*&
_output_shapes
:*
dtype0
r
stream_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namestream_2/bias
k
!stream_2/bias/Read/ReadVariableOpReadVariableOpstream_2/bias*
_output_shapes
:*
dtype0
�
stream_2/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namestream_2/depthwise_kernel
�
-stream_2/depthwise_kernel/Read/ReadVariableOpReadVariableOpstream_2/depthwise_kernel*&
_output_shapes
:*
dtype0
r
stream_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namestream_1/bias
k
!stream_1/bias/Read/ReadVariableOpReadVariableOpstream_1/bias*
_output_shapes
:*
dtype0
�
stream_1/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namestream_1/depthwise_kernel
�
-stream_1/depthwise_kernel/Read/ReadVariableOpReadVariableOpstream_1/depthwise_kernel*&
_output_shapes
:*
dtype0
n
stream/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namestream/bias
g
stream/bias/Read/ReadVariableOpReadVariableOpstream/bias*
_output_shapes
:*
dtype0
~
stream/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namestream/kernel
w
!stream/kernel/Read/ReadVariableOpReadVariableOpstream/kernel*&
_output_shapes
:*
dtype0
�
streaming/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_namestreaming/dense_1/bias
}
*streaming/dense_1/bias/Read/ReadVariableOpReadVariableOpstreaming/dense_1/bias*
_output_shapes
:*
dtype0
�
streaming/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*)
shared_namestreaming/dense_1/kernel
�
,streaming/dense_1/kernel/Read/ReadVariableOpReadVariableOpstreaming/dense_1/kernel*
_output_shapes

:
*
dtype0
�
streaming/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*%
shared_namestreaming/dense/bias
y
(streaming/dense/bias/Read/ReadVariableOpReadVariableOpstreaming/dense/bias*
_output_shapes
:
*
dtype0
�
streaming/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:0
*'
shared_namestreaming/dense/kernel
�
*streaming/dense/kernel/Read/ReadVariableOpReadVariableOpstreaming/dense/kernel*
_output_shapes

:0
*
dtype0
�
streaming/stream_6/statesVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namestreaming/stream_6/states
�
-streaming/stream_6/states/Read/ReadVariableOpReadVariableOpstreaming/stream_6/states*&
_output_shapes
:*
dtype0
�
/streaming/batch_normalization_4/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*@
shared_name1/streaming/batch_normalization_4/moving_variance
�
Cstreaming/batch_normalization_4/moving_variance/Read/ReadVariableOpReadVariableOp/streaming/batch_normalization_4/moving_variance*
_output_shapes
:*
dtype0
�
+streaming/batch_normalization_4/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*<
shared_name-+streaming/batch_normalization_4/moving_mean
�
?streaming/batch_normalization_4/moving_mean/Read/ReadVariableOpReadVariableOp+streaming/batch_normalization_4/moving_mean*
_output_shapes
:*
dtype0
�
$streaming/batch_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$streaming/batch_normalization_4/beta
�
8streaming/batch_normalization_4/beta/Read/ReadVariableOpReadVariableOp$streaming/batch_normalization_4/beta*
_output_shapes
:*
dtype0
�
%streaming/batch_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%streaming/batch_normalization_4/gamma
�
9streaming/batch_normalization_4/gamma/Read/ReadVariableOpReadVariableOp%streaming/batch_normalization_4/gamma*
_output_shapes
:*
dtype0
�
streaming/stream_5/statesVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namestreaming/stream_5/states
�
-streaming/stream_5/states/Read/ReadVariableOpReadVariableOpstreaming/stream_5/states*&
_output_shapes
:*
dtype0
�
/streaming/batch_normalization_3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*@
shared_name1/streaming/batch_normalization_3/moving_variance
�
Cstreaming/batch_normalization_3/moving_variance/Read/ReadVariableOpReadVariableOp/streaming/batch_normalization_3/moving_variance*
_output_shapes
:*
dtype0
�
+streaming/batch_normalization_3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*<
shared_name-+streaming/batch_normalization_3/moving_mean
�
?streaming/batch_normalization_3/moving_mean/Read/ReadVariableOpReadVariableOp+streaming/batch_normalization_3/moving_mean*
_output_shapes
:*
dtype0
�
$streaming/batch_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$streaming/batch_normalization_3/beta
�
8streaming/batch_normalization_3/beta/Read/ReadVariableOpReadVariableOp$streaming/batch_normalization_3/beta*
_output_shapes
:*
dtype0
�
%streaming/batch_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%streaming/batch_normalization_3/gamma
�
9streaming/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOp%streaming/batch_normalization_3/gamma*
_output_shapes
:*
dtype0
�
streaming/stream_4/statesVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namestreaming/stream_4/states
�
-streaming/stream_4/states/Read/ReadVariableOpReadVariableOpstreaming/stream_4/states*&
_output_shapes
:*
dtype0
�
streaming/stream_3/statesVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namestreaming/stream_3/states
�
-streaming/stream_3/states/Read/ReadVariableOpReadVariableOpstreaming/stream_3/states*&
_output_shapes
:*
dtype0
�
/streaming/batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*@
shared_name1/streaming/batch_normalization_2/moving_variance
�
Cstreaming/batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp/streaming/batch_normalization_2/moving_variance*
_output_shapes
:*
dtype0
�
+streaming/batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*<
shared_name-+streaming/batch_normalization_2/moving_mean
�
?streaming/batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp+streaming/batch_normalization_2/moving_mean*
_output_shapes
:*
dtype0
�
$streaming/batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$streaming/batch_normalization_2/beta
�
8streaming/batch_normalization_2/beta/Read/ReadVariableOpReadVariableOp$streaming/batch_normalization_2/beta*
_output_shapes
:*
dtype0
�
%streaming/batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%streaming/batch_normalization_2/gamma
�
9streaming/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOp%streaming/batch_normalization_2/gamma*
_output_shapes
:*
dtype0
�
streaming/stream_2/statesVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namestreaming/stream_2/states
�
-streaming/stream_2/states/Read/ReadVariableOpReadVariableOpstreaming/stream_2/states*&
_output_shapes
:*
dtype0
�
/streaming/batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*@
shared_name1/streaming/batch_normalization_1/moving_variance
�
Cstreaming/batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp/streaming/batch_normalization_1/moving_variance*
_output_shapes
:*
dtype0
�
+streaming/batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*<
shared_name-+streaming/batch_normalization_1/moving_mean
�
?streaming/batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp+streaming/batch_normalization_1/moving_mean*
_output_shapes
:*
dtype0
�
$streaming/batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$streaming/batch_normalization_1/beta
�
8streaming/batch_normalization_1/beta/Read/ReadVariableOpReadVariableOp$streaming/batch_normalization_1/beta*
_output_shapes
:*
dtype0
�
%streaming/batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%streaming/batch_normalization_1/gamma
�
9streaming/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOp%streaming/batch_normalization_1/gamma*
_output_shapes
:*
dtype0
�
streaming/stream_1/statesVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namestreaming/stream_1/states
�
-streaming/stream_1/states/Read/ReadVariableOpReadVariableOpstreaming/stream_1/states*&
_output_shapes
:*
dtype0
�
-streaming/batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*>
shared_name/-streaming/batch_normalization/moving_variance
�
Astreaming/batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp-streaming/batch_normalization/moving_variance*
_output_shapes
:*
dtype0
�
)streaming/batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*:
shared_name+)streaming/batch_normalization/moving_mean
�
=streaming/batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOp)streaming/batch_normalization/moving_mean*
_output_shapes
:*
dtype0
�
"streaming/batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"streaming/batch_normalization/beta
�
6streaming/batch_normalization/beta/Read/ReadVariableOpReadVariableOp"streaming/batch_normalization/beta*
_output_shapes
:*
dtype0
�
#streaming/batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#streaming/batch_normalization/gamma
�
7streaming/batch_normalization/gamma/Read/ReadVariableOpReadVariableOp#streaming/batch_normalization/gamma*
_output_shapes
:*
dtype0
�
streaming/stream/statesVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_namestreaming/stream/states
�
+streaming/stream/states/Read/ReadVariableOpReadVariableOpstreaming/stream/states*&
_output_shapes
:*
dtype0
}
serving_default_IEGM_inputPlaceholder*'
_output_shapes
:�*
dtype0*
shape:�
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_IEGM_inputstreaming/stream/statesstream/kernelstream/bias#streaming/batch_normalization/gamma"streaming/batch_normalization/beta)streaming/batch_normalization/moving_mean-streaming/batch_normalization/moving_variancestreaming/stream_1/statesstream_1/depthwise_kernelstream_1/bias%streaming/batch_normalization_1/gamma$streaming/batch_normalization_1/beta+streaming/batch_normalization_1/moving_mean/streaming/batch_normalization_1/moving_variancestreaming/stream_2/statesstream_2/depthwise_kernelstream_2/bias%streaming/batch_normalization_2/gamma$streaming/batch_normalization_2/beta+streaming/batch_normalization_2/moving_mean/streaming/batch_normalization_2/moving_variancestreaming/stream_3/statesstreaming/stream_4/statesstream_4/depthwise_kernelstream_4/bias%streaming/batch_normalization_3/gamma$streaming/batch_normalization_3/beta+streaming/batch_normalization_3/moving_mean/streaming/batch_normalization_3/moving_variancestreaming/stream_5/statesstream_5/depthwise_kernelstream_5/bias%streaming/batch_normalization_4/gamma$streaming/batch_normalization_4/beta+streaming/batch_normalization_4/moving_mean/streaming/batch_normalization_4/moving_variancestreaming/stream_6/statesstreaming/dense/kernelstreaming/dense/biasstreaming/dense_1/kernelstreaming/dense_1/bias*5
Tin.
,2**
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*D
_read_only_resource_inputs&
$"	
 !"#$&'()*-
config_proto

CPU

GPU 2J 8� *,
f'R%
#__inference_signature_wrapper_11841

NoOpNoOp
Ե
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*��
value��B�� B��
�
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer_with_weights-5

layer-9
layer-10
layer_with_weights-6
layer-11
layer_with_weights-7
layer-12
layer_with_weights-8
layer-13
layer-14
layer_with_weights-9
layer-15
layer_with_weights-10
layer-16
layer-17
layer-18
layer_with_weights-11
layer-19
layer_with_weights-12
layer-20
layer_with_weights-13
layer-21
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
* 
�
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses* 
�
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses
+cell
,state_shape

-states*
�
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses
4axis
	5gamma
6beta
7moving_mean
8moving_variance*
�
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses* 
�
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
C__call__
*D&call_and_return_all_conditional_losses
Ecell
Fstate_shape

Gstates*
�
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
L__call__
*M&call_and_return_all_conditional_losses
Naxis
	Ogamma
Pbeta
Qmoving_mean
Rmoving_variance*
�
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W__call__
*X&call_and_return_all_conditional_losses* 
�
Y	variables
Ztrainable_variables
[regularization_losses
\	keras_api
]__call__
*^&call_and_return_all_conditional_losses
_cell
`state_shape

astates*
�
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
f__call__
*g&call_and_return_all_conditional_losses
haxis
	igamma
jbeta
kmoving_mean
lmoving_variance*
�
m	variables
ntrainable_variables
oregularization_losses
p	keras_api
q__call__
*r&call_and_return_all_conditional_losses* 
�
s	variables
ttrainable_variables
uregularization_losses
v	keras_api
w__call__
*x&call_and_return_all_conditional_losses
ycell
zstate_shape

{states*
�
|	variables
}trainable_variables
~regularization_losses
	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�cell
�state_shape
�states*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�cell
�state_shape
�states*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�cell
�state_shape
�states*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias*
�
�0
�1
-2
53
64
75
86
�7
�8
G9
O10
P11
Q12
R13
�14
�15
a16
i17
j18
k19
l20
{21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40*
�
�0
�1
52
63
�4
�5
O6
P7
�8
�9
i10
j11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
:
�trace_0
�trace_1
�trace_2
�trace_3* 
:
�trace_0
�trace_1
�trace_2
�trace_3* 
* 

�serving_default* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

�0
�1
-2*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
* 
ga
VARIABLE_VALUEstreaming/stream/states6layer_with_weights-0/states/.ATTRIBUTES/VARIABLE_VALUE*
 
50
61
72
83*

50
61*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
rl
VARIABLE_VALUE#streaming/batch_normalization/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUE"streaming/batch_normalization/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE)streaming/batch_normalization/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE-streaming/batch_normalization/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

�0
�1
G2*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
?	variables
@trainable_variables
Aregularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�depthwise_kernel
	�bias
!�_jit_compiled_convolution_op*
* 
ic
VARIABLE_VALUEstreaming/stream_1/states6layer_with_weights-2/states/.ATTRIBUTES/VARIABLE_VALUE*
 
O0
P1
Q2
R3*

O0
P1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
H	variables
Itrainable_variables
Jregularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
tn
VARIABLE_VALUE%streaming/batch_normalization_1/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE$streaming/batch_normalization_1/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE*
�z
VARIABLE_VALUE+streaming/batch_normalization_1/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE/streaming/batch_normalization_1/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

�0
�1
a2*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
Y	variables
Ztrainable_variables
[regularization_losses
]__call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�depthwise_kernel
	�bias
!�_jit_compiled_convolution_op*
* 
ic
VARIABLE_VALUEstreaming/stream_2/states6layer_with_weights-4/states/.ATTRIBUTES/VARIABLE_VALUE*
 
i0
j1
k2
l3*

i0
j1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
b	variables
ctrainable_variables
dregularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
tn
VARIABLE_VALUE%streaming/batch_normalization_2/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE$streaming/batch_normalization_2/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE*
�z
VARIABLE_VALUE+streaming/batch_normalization_2/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE/streaming/batch_normalization_2/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
m	variables
ntrainable_variables
oregularization_losses
q__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

{0*
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
s	variables
ttrainable_variables
uregularization_losses
w__call__
*x&call_and_return_all_conditional_losses
&x"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
* 
ic
VARIABLE_VALUEstreaming/stream_3/states6layer_with_weights-6/states/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1
�2*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
|	variables
}trainable_variables
~regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�depthwise_kernel
	�bias
!�_jit_compiled_convolution_op*
* 
ic
VARIABLE_VALUEstreaming/stream_4/states6layer_with_weights-7/states/.ATTRIBUTES/VARIABLE_VALUE*
$
�0
�1
�2
�3*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
tn
VARIABLE_VALUE%streaming/batch_normalization_3/gamma5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE$streaming/batch_normalization_3/beta4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUE*
�z
VARIABLE_VALUE+streaming/batch_normalization_3/moving_mean;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE/streaming/batch_normalization_3/moving_variance?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

�0
�1
�2*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�depthwise_kernel
	�bias
!�_jit_compiled_convolution_op*
* 
ic
VARIABLE_VALUEstreaming/stream_5/states6layer_with_weights-9/states/.ATTRIBUTES/VARIABLE_VALUE*
$
�0
�1
�2
�3*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
uo
VARIABLE_VALUE%streaming/batch_normalization_4/gamma6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUE$streaming/batch_normalization_4/beta5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUE*
�{
VARIABLE_VALUE+streaming/batch_normalization_4/moving_mean<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE/streaming/batch_normalization_4/moving_variance@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

�0*
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
* 
jd
VARIABLE_VALUEstreaming/stream_6/states7layer_with_weights-11/states/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
ga
VARIABLE_VALUEstreaming/dense/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEstreaming/dense/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
ic
VARIABLE_VALUEstreaming/dense_1/kernel7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEstreaming/dense_1/bias5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEstream/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUEstream/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEstream_1/depthwise_kernel&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEstream_1/bias&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEstream_2/depthwise_kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEstream_2/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEstream_4/depthwise_kernel'variables/22/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEstream_4/bias'variables/23/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEstream_5/depthwise_kernel'variables/29/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEstream_5/bias'variables/30/.ATTRIBUTES/VARIABLE_VALUE*
�
-0
71
82
G3
Q4
R5
a6
k7
l8
{9
�10
�11
�12
�13
�14
�15
�16*
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

-0*

+0*
* 
* 
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 

70
81*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

G0*

E0*
* 
* 
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 

Q0
R1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

a0*

_0*
* 
* 
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 

k0
l1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

{0*
	
y0* 
* 
* 
* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

�0*

�0*
* 
* 
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 

�0
�1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

�0*

�0*
* 
* 
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 

�0
�1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

�0*


�0* 
* 
* 
* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamestreaming/stream/states#streaming/batch_normalization/gamma"streaming/batch_normalization/beta)streaming/batch_normalization/moving_mean-streaming/batch_normalization/moving_variancestreaming/stream_1/states%streaming/batch_normalization_1/gamma$streaming/batch_normalization_1/beta+streaming/batch_normalization_1/moving_mean/streaming/batch_normalization_1/moving_variancestreaming/stream_2/states%streaming/batch_normalization_2/gamma$streaming/batch_normalization_2/beta+streaming/batch_normalization_2/moving_mean/streaming/batch_normalization_2/moving_variancestreaming/stream_3/statesstreaming/stream_4/states%streaming/batch_normalization_3/gamma$streaming/batch_normalization_3/beta+streaming/batch_normalization_3/moving_mean/streaming/batch_normalization_3/moving_variancestreaming/stream_5/states%streaming/batch_normalization_4/gamma$streaming/batch_normalization_4/beta+streaming/batch_normalization_4/moving_mean/streaming/batch_normalization_4/moving_variancestreaming/stream_6/statesstreaming/dense/kernelstreaming/dense/biasstreaming/dense_1/kernelstreaming/dense_1/biasstream/kernelstream/biasstream_1/depthwise_kernelstream_1/biasstream_2/depthwise_kernelstream_2/biasstream_4/depthwise_kernelstream_4/biasstream_5/depthwise_kernelstream_5/biasConst*6
Tin/
-2+*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *'
f"R 
__inference__traced_save_13173
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamestreaming/stream/states#streaming/batch_normalization/gamma"streaming/batch_normalization/beta)streaming/batch_normalization/moving_mean-streaming/batch_normalization/moving_variancestreaming/stream_1/states%streaming/batch_normalization_1/gamma$streaming/batch_normalization_1/beta+streaming/batch_normalization_1/moving_mean/streaming/batch_normalization_1/moving_variancestreaming/stream_2/states%streaming/batch_normalization_2/gamma$streaming/batch_normalization_2/beta+streaming/batch_normalization_2/moving_mean/streaming/batch_normalization_2/moving_variancestreaming/stream_3/statesstreaming/stream_4/states%streaming/batch_normalization_3/gamma$streaming/batch_normalization_3/beta+streaming/batch_normalization_3/moving_mean/streaming/batch_normalization_3/moving_variancestreaming/stream_5/states%streaming/batch_normalization_4/gamma$streaming/batch_normalization_4/beta+streaming/batch_normalization_4/moving_mean/streaming/batch_normalization_4/moving_variancestreaming/stream_6/statesstreaming/dense/kernelstreaming/dense/biasstreaming/dense_1/kernelstreaming/dense_1/biasstream/kernelstream/biasstream_1/depthwise_kernelstream_1/biasstream_2/depthwise_kernelstream_2/biasstream_4/depthwise_kernelstream_4/biasstream_5/depthwise_kernelstream_5/bias*5
Tin.
,2**
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__traced_restore_13306�
�

�
5__inference_batch_normalization_1_layer_call_fn_12461

inputs3
%streaming_batch_normalization_1_gamma:2
$streaming_batch_normalization_1_beta:9
+streaming_batch_normalization_1_moving_mean:=
/streaming_batch_normalization_1_moving_variance:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs%streaming_batch_normalization_1_gamma$streaming_batch_normalization_1_beta+streaming_batch_normalization_1_moving_mean/streaming_batch_normalization_1_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_10375�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
H
,__inference_activation_4_layer_call_fn_12806

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_activation_4_layer_call_and_return_conditional_losses_10904_
IdentityIdentityPartitionedCall:output:0*
T0*&
_output_shapes
:"
identityIdentity:output:0*%
_input_shapes
::N J
&
_output_shapes
:
 
_user_specified_nameinputs
�

�
5__inference_batch_normalization_2_layer_call_fn_12562

inputs3
%streaming_batch_normalization_2_gamma:2
$streaming_batch_normalization_2_beta:9
+streaming_batch_normalization_2_moving_mean:=
/streaming_batch_normalization_2_moving_variance:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs%streaming_batch_normalization_2_gamma$streaming_batch_normalization_2_beta+streaming_batch_normalization_2_moving_mean/streaming_batch_normalization_2_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_10494�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
I
-__inference_max_pooling2d_layer_call_fn_12329

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_10222�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
d
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_10222

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
N__inference_batch_normalization_layer_call_and_return_conditional_losses_12396

inputs@
2readvariableop_streaming_batch_normalization_gamma:A
3readvariableop_1_streaming_batch_normalization_beta:W
Ifusedbatchnormv3_readvariableop_streaming_batch_normalization_moving_mean:]
Ofusedbatchnormv3_readvariableop_1_streaming_batch_normalization_moving_variance:
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1}
ReadVariableOpReadVariableOp2readvariableop_streaming_batch_normalization_gamma*
_output_shapes
:*
dtype0�
ReadVariableOp_1ReadVariableOp3readvariableop_1_streaming_batch_normalization_beta*
_output_shapes
:*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOpIfusedbatchnormv3_readvariableop_streaming_batch_normalization_moving_mean*
_output_shapes
:*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOfusedbatchnormv3_readvariableop_1_streaming_batch_normalization_moving_variance*
_output_shapes
:*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%��'7*
exponential_avg_factor%fff?�
AssignNewValueAssignVariableOpIfusedbatchnormv3_readvariableop_streaming_batch_normalization_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOpOfusedbatchnormv3_readvariableop_1_streaming_batch_normalization_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_10603

inputsB
4readvariableop_streaming_batch_normalization_3_gamma:C
5readvariableop_1_streaming_batch_normalization_3_beta:Y
Kfusedbatchnormv3_readvariableop_streaming_batch_normalization_3_moving_mean:_
Qfusedbatchnormv3_readvariableop_1_streaming_batch_normalization_3_moving_variance:
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1
ReadVariableOpReadVariableOp4readvariableop_streaming_batch_normalization_3_gamma*
_output_shapes
:*
dtype0�
ReadVariableOp_1ReadVariableOp5readvariableop_1_streaming_batch_normalization_3_beta*
_output_shapes
:*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOpKfusedbatchnormv3_readvariableop_streaming_batch_normalization_3_moving_mean*
_output_shapes
:*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpQfusedbatchnormv3_readvariableop_1_streaming_batch_normalization_3_moving_variance*
_output_shapes
:*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%��'7*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�

�
3__inference_batch_normalization_layer_call_fn_12378

inputs1
#streaming_batch_normalization_gamma:0
"streaming_batch_normalization_beta:7
)streaming_batch_normalization_moving_mean:;
-streaming_batch_normalization_moving_variance:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs#streaming_batch_normalization_gamma"streaming_batch_normalization_beta)streaming_batch_normalization_moving_mean-streaming_batch_normalization_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_10310�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�

�
5__inference_batch_normalization_4_layer_call_fn_12765

inputs3
%streaming_batch_normalization_4_gamma:2
$streaming_batch_normalization_4_beta:9
+streaming_batch_normalization_4_moving_mean:=
/streaming_batch_normalization_4_moving_variance:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs%streaming_batch_normalization_4_gamma$streaming_batch_normalization_4_beta+streaming_batch_normalization_4_moving_mean/streaming_batch_normalization_4_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_10695�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
(__inference_stream_4_layer_call_fn_12635

inputs3
streaming_stream_4_states:3
stream_4_depthwise_kernel:
stream_4_bias:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstreaming_stream_4_statesstream_4_depthwise_kernelstream_4_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_stream_4_layer_call_and_return_conditional_losses_10852n
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*&
_output_shapes
:`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*+
_input_shapes
:: : : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
:
 
_user_specified_nameinputs
�!
�
%__inference_model_layer_call_fn_11152

iegm_input1
streaming_stream_states:'
stream_kernel:
stream_bias:1
#streaming_batch_normalization_gamma:0
"streaming_batch_normalization_beta:7
)streaming_batch_normalization_moving_mean:;
-streaming_batch_normalization_moving_variance:3
streaming_stream_1_states:3
stream_1_depthwise_kernel:
stream_1_bias:3
%streaming_batch_normalization_1_gamma:2
$streaming_batch_normalization_1_beta:9
+streaming_batch_normalization_1_moving_mean:=
/streaming_batch_normalization_1_moving_variance:3
streaming_stream_2_states:3
stream_2_depthwise_kernel:
stream_2_bias:3
%streaming_batch_normalization_2_gamma:2
$streaming_batch_normalization_2_beta:9
+streaming_batch_normalization_2_moving_mean:=
/streaming_batch_normalization_2_moving_variance:3
streaming_stream_3_states:3
streaming_stream_4_states:3
stream_4_depthwise_kernel:
stream_4_bias:3
%streaming_batch_normalization_3_gamma:2
$streaming_batch_normalization_3_beta:9
+streaming_batch_normalization_3_moving_mean:=
/streaming_batch_normalization_3_moving_variance:3
streaming_stream_5_states:3
stream_5_depthwise_kernel:
stream_5_bias:3
%streaming_batch_normalization_4_gamma:2
$streaming_batch_normalization_4_beta:9
+streaming_batch_normalization_4_moving_mean:=
/streaming_batch_normalization_4_moving_variance:3
streaming_stream_6_states:(
streaming_dense_kernel:0
"
streaming_dense_bias:
*
streaming_dense_1_kernel:
$
streaming_dense_1_bias:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCall
iegm_inputstreaming_stream_statesstream_kernelstream_bias#streaming_batch_normalization_gamma"streaming_batch_normalization_beta)streaming_batch_normalization_moving_mean-streaming_batch_normalization_moving_variancestreaming_stream_1_statesstream_1_depthwise_kernelstream_1_bias%streaming_batch_normalization_1_gamma$streaming_batch_normalization_1_beta+streaming_batch_normalization_1_moving_mean/streaming_batch_normalization_1_moving_variancestreaming_stream_2_statesstream_2_depthwise_kernelstream_2_bias%streaming_batch_normalization_2_gamma$streaming_batch_normalization_2_beta+streaming_batch_normalization_2_moving_mean/streaming_batch_normalization_2_moving_variancestreaming_stream_3_statesstreaming_stream_4_statesstream_4_depthwise_kernelstream_4_bias%streaming_batch_normalization_3_gamma$streaming_batch_normalization_3_beta+streaming_batch_normalization_3_moving_mean/streaming_batch_normalization_3_moving_variancestreaming_stream_5_statesstream_5_depthwise_kernelstream_5_bias%streaming_batch_normalization_4_gamma$streaming_batch_normalization_4_beta+streaming_batch_normalization_4_moving_mean/streaming_batch_normalization_4_moving_variancestreaming_stream_6_statesstreaming_dense_kernelstreaming_dense_biasstreaming_dense_1_kernelstreaming_dense_1_bias*5
Tin.
,2**
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*:
_read_only_resource_inputs
	
 !"&'()*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_11108f
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*x
_input_shapesg
e:�: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
'
_output_shapes
:�
$
_user_specified_name
IEGM_input
�
�
(__inference_stream_1_layer_call_fn_12432

inputs3
streaming_stream_1_states:3
stream_1_depthwise_kernel:
stream_1_bias:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstreaming_stream_1_statesstream_1_depthwise_kernelstream_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_stream_1_layer_call_and_return_conditional_losses_10762n
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*&
_output_shapes
:`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*+
_input_shapes
:$: : : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
:$
 
_user_specified_nameinputs
�
�
C__inference_stream_3_layer_call_and_return_conditional_losses_12627

inputsI
/concat_readvariableop_streaming_stream_3_states:
identity��AssignVariableOp�concat/ReadVariableOp�
concat/ReadVariableOpReadVariableOp/concat_readvariableop_streaming_stream_3_states*&
_output_shapes
:*
dtype0M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatConcatV2concat/ReadVariableOp:value:0inputsconcat/axis:output:0*
N*
T0*&
_output_shapes
:h
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ����    j
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            j
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         �
strided_sliceStridedSliceconcat:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*&
_output_shapes
:*

begin_mask*
end_mask�
AssignVariableOpAssignVariableOp/concat_readvariableop_streaming_stream_3_statesstrided_slice:output:0^concat/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
average_pooling2d/AvgPoolAvgPoolconcat:output:0^AssignVariableOp*
T0*&
_output_shapes
:*
ksize
*
paddingVALID*
strides
p
IdentityIdentity"average_pooling2d/AvgPool:output:0^NoOp*
T0*&
_output_shapes
:q
NoOpNoOp^AssignVariableOp^concat/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*'
_input_shapes
:: 2$
AssignVariableOpAssignVariableOp2.
concat/ReadVariableOpconcat/ReadVariableOp:N J
&
_output_shapes
:
 
_user_specified_nameinputs
�
c
G__inference_activation_2_layer_call_and_return_conditional_losses_12608

inputs
identityE
ReluReluinputs*
T0*&
_output_shapes
:Y
IdentityIdentityRelu:activations:0*
T0*&
_output_shapes
:"
identityIdentity:output:0*%
_input_shapes
::N J
&
_output_shapes
:
 
_user_specified_nameinputs
�
�
C__inference_stream_3_layer_call_and_return_conditional_losses_10829

inputsI
/concat_readvariableop_streaming_stream_3_states:
identity��AssignVariableOp�concat/ReadVariableOp�
concat/ReadVariableOpReadVariableOp/concat_readvariableop_streaming_stream_3_states*&
_output_shapes
:*
dtype0M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatConcatV2concat/ReadVariableOp:value:0inputsconcat/axis:output:0*
N*
T0*&
_output_shapes
:h
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ����    j
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            j
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         �
strided_sliceStridedSliceconcat:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*&
_output_shapes
:*

begin_mask*
end_mask�
AssignVariableOpAssignVariableOp/concat_readvariableop_streaming_stream_3_statesstrided_slice:output:0^concat/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
average_pooling2d/AvgPoolAvgPoolconcat:output:0^AssignVariableOp*
T0*&
_output_shapes
:*
ksize
*
paddingVALID*
strides
p
IdentityIdentity"average_pooling2d/AvgPool:output:0^NoOp*
T0*&
_output_shapes
:q
NoOpNoOp^AssignVariableOp^concat/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:: 2$
AssignVariableOpAssignVariableOp2.
concat/ReadVariableOpconcat/ReadVariableOp:N J
&
_output_shapes
:
 
_user_specified_nameinputs
�
h
L__inference_average_pooling2d_layer_call_and_return_conditional_losses_10515

inputs
identity�
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
`
B__inference_dropout_layer_call_and_return_conditional_losses_11028

inputs

identity_1M
IdentityIdentityinputs*
T0*&
_output_shapes
:Z

Identity_1IdentityIdentity:output:0*
T0*&
_output_shapes
:"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
::N J
&
_output_shapes
:
 
_user_specified_nameinputs
�
�
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_12598

inputsB
4readvariableop_streaming_batch_normalization_2_gamma:C
5readvariableop_1_streaming_batch_normalization_2_beta:Y
Kfusedbatchnormv3_readvariableop_streaming_batch_normalization_2_moving_mean:_
Qfusedbatchnormv3_readvariableop_1_streaming_batch_normalization_2_moving_variance:
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1
ReadVariableOpReadVariableOp4readvariableop_streaming_batch_normalization_2_gamma*
_output_shapes
:*
dtype0�
ReadVariableOp_1ReadVariableOp5readvariableop_1_streaming_batch_normalization_2_beta*
_output_shapes
:*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOpKfusedbatchnormv3_readvariableop_streaming_batch_normalization_2_moving_mean*
_output_shapes
:*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpQfusedbatchnormv3_readvariableop_1_streaming_batch_normalization_2_moving_variance*
_output_shapes
:*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%��'7*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�w
�
@__inference_model_layer_call_and_return_conditional_losses_11108

inputs8
stream_streaming_stream_states:.
stream_stream_kernel: 
stream_stream_bias:E
7batch_normalization_streaming_batch_normalization_gamma:D
6batch_normalization_streaming_batch_normalization_beta:K
=batch_normalization_streaming_batch_normalization_moving_mean:O
Abatch_normalization_streaming_batch_normalization_moving_variance:<
"stream_1_streaming_stream_1_states:<
"stream_1_stream_1_depthwise_kernel:$
stream_1_stream_1_bias:I
;batch_normalization_1_streaming_batch_normalization_1_gamma:H
:batch_normalization_1_streaming_batch_normalization_1_beta:O
Abatch_normalization_1_streaming_batch_normalization_1_moving_mean:S
Ebatch_normalization_1_streaming_batch_normalization_1_moving_variance:<
"stream_2_streaming_stream_2_states:<
"stream_2_stream_2_depthwise_kernel:$
stream_2_stream_2_bias:I
;batch_normalization_2_streaming_batch_normalization_2_gamma:H
:batch_normalization_2_streaming_batch_normalization_2_beta:O
Abatch_normalization_2_streaming_batch_normalization_2_moving_mean:S
Ebatch_normalization_2_streaming_batch_normalization_2_moving_variance:<
"stream_3_streaming_stream_3_states:<
"stream_4_streaming_stream_4_states:<
"stream_4_stream_4_depthwise_kernel:$
stream_4_stream_4_bias:I
;batch_normalization_3_streaming_batch_normalization_3_gamma:H
:batch_normalization_3_streaming_batch_normalization_3_beta:O
Abatch_normalization_3_streaming_batch_normalization_3_moving_mean:S
Ebatch_normalization_3_streaming_batch_normalization_3_moving_variance:<
"stream_5_streaming_stream_5_states:<
"stream_5_stream_5_depthwise_kernel:$
stream_5_stream_5_bias:I
;batch_normalization_4_streaming_batch_normalization_4_gamma:H
:batch_normalization_4_streaming_batch_normalization_4_beta:O
Abatch_normalization_4_streaming_batch_normalization_4_moving_mean:S
Ebatch_normalization_4_streaming_batch_normalization_4_moving_variance:<
"stream_6_streaming_stream_6_states:.
dense_streaming_dense_kernel:0
(
dense_streaming_dense_bias:
2
 dense_1_streaming_dense_1_kernel:
,
dense_1_streaming_dense_1_bias:
identity��+batch_normalization/StatefulPartitionedCall�-batch_normalization_1/StatefulPartitionedCall�-batch_normalization_2/StatefulPartitionedCall�-batch_normalization_3/StatefulPartitionedCall�-batch_normalization_4/StatefulPartitionedCall�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dropout/StatefulPartitionedCall�stream/StatefulPartitionedCall� stream_1/StatefulPartitionedCall� stream_2/StatefulPartitionedCall� stream_3/StatefulPartitionedCall� stream_4/StatefulPartitionedCall� stream_5/StatefulPartitionedCall� stream_6/StatefulPartitionedCall�
max_pooling2d/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:H* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_10222�
stream/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0stream_streaming_stream_statesstream_stream_kernelstream_stream_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_stream_layer_call_and_return_conditional_losses_10725�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'stream/StatefulPartitionedCall:output:07batch_normalization_streaming_batch_normalization_gamma6batch_normalization_streaming_batch_normalization_beta=batch_normalization_streaming_batch_normalization_moving_meanAbatch_normalization_streaming_batch_normalization_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_10283�
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:$* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_10740�
 stream_1/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0"stream_1_streaming_stream_1_states"stream_1_stream_1_depthwise_kernelstream_1_stream_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_stream_1_layer_call_and_return_conditional_losses_10762�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)stream_1/StatefulPartitionedCall:output:0;batch_normalization_1_streaming_batch_normalization_1_gamma:batch_normalization_1_streaming_batch_normalization_1_betaAbatch_normalization_1_streaming_batch_normalization_1_moving_meanEbatch_normalization_1_streaming_batch_normalization_1_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_10375�
activation_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_10777�
 stream_2/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0"stream_2_streaming_stream_2_states"stream_2_stream_2_depthwise_kernelstream_2_stream_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_stream_2_layer_call_and_return_conditional_losses_10799�
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)stream_2/StatefulPartitionedCall:output:0;batch_normalization_2_streaming_batch_normalization_2_gamma:batch_normalization_2_streaming_batch_normalization_2_betaAbatch_normalization_2_streaming_batch_normalization_2_moving_meanEbatch_normalization_2_streaming_batch_normalization_2_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_10467�
activation_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_activation_2_layer_call_and_return_conditional_losses_10814�
 stream_3/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:0"stream_3_streaming_stream_3_states*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_stream_3_layer_call_and_return_conditional_losses_10829�
 stream_4/StatefulPartitionedCallStatefulPartitionedCall)stream_3/StatefulPartitionedCall:output:0"stream_4_streaming_stream_4_states"stream_4_stream_4_depthwise_kernelstream_4_stream_4_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_stream_4_layer_call_and_return_conditional_losses_10852�
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)stream_4/StatefulPartitionedCall:output:0;batch_normalization_3_streaming_batch_normalization_3_gamma:batch_normalization_3_streaming_batch_normalization_3_betaAbatch_normalization_3_streaming_batch_normalization_3_moving_meanEbatch_normalization_3_streaming_batch_normalization_3_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_10576�
activation_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_activation_3_layer_call_and_return_conditional_losses_10867�
 stream_5/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0"stream_5_streaming_stream_5_states"stream_5_stream_5_depthwise_kernelstream_5_stream_5_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_stream_5_layer_call_and_return_conditional_losses_10889�
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall)stream_5/StatefulPartitionedCall:output:0;batch_normalization_4_streaming_batch_normalization_4_gamma:batch_normalization_4_streaming_batch_normalization_4_betaAbatch_normalization_4_streaming_batch_normalization_4_moving_meanEbatch_normalization_4_streaming_batch_normalization_4_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_10668�
activation_4/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_activation_4_layer_call_and_return_conditional_losses_10904�
dropout/StatefulPartitionedCallStatefulPartitionedCall%activation_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_10918�
 stream_6/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0"stream_6_streaming_stream_6_states*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_stream_6_layer_call_and_return_conditional_losses_10934�
dense/StatefulPartitionedCallStatefulPartitionedCall)stream_6/StatefulPartitionedCall:output:0dense_streaming_dense_kerneldense_streaming_dense_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_10948�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0 dense_1_streaming_dense_1_kerneldense_1_streaming_dense_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_10963n
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:�
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall^stream/StatefulPartitionedCall!^stream_1/StatefulPartitionedCall!^stream_2/StatefulPartitionedCall!^stream_3/StatefulPartitionedCall!^stream_4/StatefulPartitionedCall!^stream_5/StatefulPartitionedCall!^stream_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*x
_input_shapesg
e:�: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2@
stream/StatefulPartitionedCallstream/StatefulPartitionedCall2D
 stream_1/StatefulPartitionedCall stream_1/StatefulPartitionedCall2D
 stream_2/StatefulPartitionedCall stream_2/StatefulPartitionedCall2D
 stream_3/StatefulPartitionedCall stream_3/StatefulPartitionedCall2D
 stream_4/StatefulPartitionedCall stream_4/StatefulPartitionedCall2D
 stream_5/StatefulPartitionedCall stream_5/StatefulPartitionedCall2D
 stream_6/StatefulPartitionedCall stream_6/StatefulPartitionedCall:O K
'
_output_shapes
:�
 
_user_specified_nameinputs
�
�
(__inference_stream_5_layer_call_fn_12727

inputs3
streaming_stream_5_states:3
stream_5_depthwise_kernel:
stream_5_bias:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstreaming_stream_5_statesstream_5_depthwise_kernelstream_5_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_stream_5_layer_call_and_return_conditional_losses_10889n
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*&
_output_shapes
:`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*+
_input_shapes
:: : : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
:
 
_user_specified_nameinputs
�
�
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_10375

inputsB
4readvariableop_streaming_batch_normalization_1_gamma:C
5readvariableop_1_streaming_batch_normalization_1_beta:Y
Kfusedbatchnormv3_readvariableop_streaming_batch_normalization_1_moving_mean:_
Qfusedbatchnormv3_readvariableop_1_streaming_batch_normalization_1_moving_variance:
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1
ReadVariableOpReadVariableOp4readvariableop_streaming_batch_normalization_1_gamma*
_output_shapes
:*
dtype0�
ReadVariableOp_1ReadVariableOp5readvariableop_1_streaming_batch_normalization_1_beta*
_output_shapes
:*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOpKfusedbatchnormv3_readvariableop_streaming_batch_normalization_1_moving_mean*
_output_shapes
:*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpQfusedbatchnormv3_readvariableop_1_streaming_batch_normalization_1_moving_variance*
_output_shapes
:*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%��'7*
exponential_avg_factor%fff?�
AssignNewValueAssignVariableOpKfusedbatchnormv3_readvariableop_streaming_batch_normalization_1_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOpQfusedbatchnormv3_readvariableop_1_streaming_batch_normalization_1_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�u
�
@__inference_model_layer_call_and_return_conditional_losses_11039

iegm_input8
stream_streaming_stream_states:.
stream_stream_kernel: 
stream_stream_bias:E
7batch_normalization_streaming_batch_normalization_gamma:D
6batch_normalization_streaming_batch_normalization_beta:K
=batch_normalization_streaming_batch_normalization_moving_mean:O
Abatch_normalization_streaming_batch_normalization_moving_variance:<
"stream_1_streaming_stream_1_states:<
"stream_1_stream_1_depthwise_kernel:$
stream_1_stream_1_bias:I
;batch_normalization_1_streaming_batch_normalization_1_gamma:H
:batch_normalization_1_streaming_batch_normalization_1_beta:O
Abatch_normalization_1_streaming_batch_normalization_1_moving_mean:S
Ebatch_normalization_1_streaming_batch_normalization_1_moving_variance:<
"stream_2_streaming_stream_2_states:<
"stream_2_stream_2_depthwise_kernel:$
stream_2_stream_2_bias:I
;batch_normalization_2_streaming_batch_normalization_2_gamma:H
:batch_normalization_2_streaming_batch_normalization_2_beta:O
Abatch_normalization_2_streaming_batch_normalization_2_moving_mean:S
Ebatch_normalization_2_streaming_batch_normalization_2_moving_variance:<
"stream_3_streaming_stream_3_states:<
"stream_4_streaming_stream_4_states:<
"stream_4_stream_4_depthwise_kernel:$
stream_4_stream_4_bias:I
;batch_normalization_3_streaming_batch_normalization_3_gamma:H
:batch_normalization_3_streaming_batch_normalization_3_beta:O
Abatch_normalization_3_streaming_batch_normalization_3_moving_mean:S
Ebatch_normalization_3_streaming_batch_normalization_3_moving_variance:<
"stream_5_streaming_stream_5_states:<
"stream_5_stream_5_depthwise_kernel:$
stream_5_stream_5_bias:I
;batch_normalization_4_streaming_batch_normalization_4_gamma:H
:batch_normalization_4_streaming_batch_normalization_4_beta:O
Abatch_normalization_4_streaming_batch_normalization_4_moving_mean:S
Ebatch_normalization_4_streaming_batch_normalization_4_moving_variance:<
"stream_6_streaming_stream_6_states:.
dense_streaming_dense_kernel:0
(
dense_streaming_dense_bias:
2
 dense_1_streaming_dense_1_kernel:
,
dense_1_streaming_dense_1_bias:
identity��+batch_normalization/StatefulPartitionedCall�-batch_normalization_1/StatefulPartitionedCall�-batch_normalization_2/StatefulPartitionedCall�-batch_normalization_3/StatefulPartitionedCall�-batch_normalization_4/StatefulPartitionedCall�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�stream/StatefulPartitionedCall� stream_1/StatefulPartitionedCall� stream_2/StatefulPartitionedCall� stream_3/StatefulPartitionedCall� stream_4/StatefulPartitionedCall� stream_5/StatefulPartitionedCall� stream_6/StatefulPartitionedCall�
max_pooling2d/PartitionedCallPartitionedCall
iegm_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:H* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_10222�
stream/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0stream_streaming_stream_statesstream_stream_kernelstream_stream_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_stream_layer_call_and_return_conditional_losses_10725�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'stream/StatefulPartitionedCall:output:07batch_normalization_streaming_batch_normalization_gamma6batch_normalization_streaming_batch_normalization_beta=batch_normalization_streaming_batch_normalization_moving_meanAbatch_normalization_streaming_batch_normalization_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:$*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_10310�
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:$* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_10740�
 stream_1/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0"stream_1_streaming_stream_1_states"stream_1_stream_1_depthwise_kernelstream_1_stream_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_stream_1_layer_call_and_return_conditional_losses_10762�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)stream_1/StatefulPartitionedCall:output:0;batch_normalization_1_streaming_batch_normalization_1_gamma:batch_normalization_1_streaming_batch_normalization_1_betaAbatch_normalization_1_streaming_batch_normalization_1_moving_meanEbatch_normalization_1_streaming_batch_normalization_1_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_10402�
activation_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_10777�
 stream_2/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0"stream_2_streaming_stream_2_states"stream_2_stream_2_depthwise_kernelstream_2_stream_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_stream_2_layer_call_and_return_conditional_losses_10799�
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)stream_2/StatefulPartitionedCall:output:0;batch_normalization_2_streaming_batch_normalization_2_gamma:batch_normalization_2_streaming_batch_normalization_2_betaAbatch_normalization_2_streaming_batch_normalization_2_moving_meanEbatch_normalization_2_streaming_batch_normalization_2_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_10494�
activation_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_activation_2_layer_call_and_return_conditional_losses_10814�
 stream_3/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:0"stream_3_streaming_stream_3_states*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_stream_3_layer_call_and_return_conditional_losses_10829�
 stream_4/StatefulPartitionedCallStatefulPartitionedCall)stream_3/StatefulPartitionedCall:output:0"stream_4_streaming_stream_4_states"stream_4_stream_4_depthwise_kernelstream_4_stream_4_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_stream_4_layer_call_and_return_conditional_losses_10852�
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)stream_4/StatefulPartitionedCall:output:0;batch_normalization_3_streaming_batch_normalization_3_gamma:batch_normalization_3_streaming_batch_normalization_3_betaAbatch_normalization_3_streaming_batch_normalization_3_moving_meanEbatch_normalization_3_streaming_batch_normalization_3_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_10603�
activation_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_activation_3_layer_call_and_return_conditional_losses_10867�
 stream_5/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0"stream_5_streaming_stream_5_states"stream_5_stream_5_depthwise_kernelstream_5_stream_5_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_stream_5_layer_call_and_return_conditional_losses_10889�
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall)stream_5/StatefulPartitionedCall:output:0;batch_normalization_4_streaming_batch_normalization_4_gamma:batch_normalization_4_streaming_batch_normalization_4_betaAbatch_normalization_4_streaming_batch_normalization_4_moving_meanEbatch_normalization_4_streaming_batch_normalization_4_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_10695�
activation_4/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_activation_4_layer_call_and_return_conditional_losses_10904�
dropout/PartitionedCallPartitionedCall%activation_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_11028�
 stream_6/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0"stream_6_streaming_stream_6_states*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_stream_6_layer_call_and_return_conditional_losses_10934�
dense/StatefulPartitionedCallStatefulPartitionedCall)stream_6/StatefulPartitionedCall:output:0dense_streaming_dense_kerneldense_streaming_dense_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_10948�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0 dense_1_streaming_dense_1_kerneldense_1_streaming_dense_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_10963n
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:�
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall^stream/StatefulPartitionedCall!^stream_1/StatefulPartitionedCall!^stream_2/StatefulPartitionedCall!^stream_3/StatefulPartitionedCall!^stream_4/StatefulPartitionedCall!^stream_5/StatefulPartitionedCall!^stream_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*x
_input_shapesg
e:�: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2@
stream/StatefulPartitionedCallstream/StatefulPartitionedCall2D
 stream_1/StatefulPartitionedCall stream_1/StatefulPartitionedCall2D
 stream_2/StatefulPartitionedCall stream_2/StatefulPartitionedCall2D
 stream_3/StatefulPartitionedCall stream_3/StatefulPartitionedCall2D
 stream_4/StatefulPartitionedCall stream_4/StatefulPartitionedCall2D
 stream_5/StatefulPartitionedCall stream_5/StatefulPartitionedCall2D
 stream_6/StatefulPartitionedCall stream_6/StatefulPartitionedCall:S O
'
_output_shapes
:�
$
_user_specified_name
IEGM_input
�
a
E__inference_activation_layer_call_and_return_conditional_losses_10740

inputs
identityE
ReluReluinputs*
T0*&
_output_shapes
:$Y
IdentityIdentityRelu:activations:0*
T0*&
_output_shapes
:$"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
:$:N J
&
_output_shapes
:$
 
_user_specified_nameinputs
�
�
C__inference_stream_6_layer_call_and_return_conditional_losses_12858

inputsB
(readvariableop_streaming_stream_6_states:
identity��AssignVariableOp�ReadVariableOp
ReadVariableOpReadVariableOp(readvariableop_streaming_stream_6_states*&
_output_shapes
:*
dtype0h
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           j
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           j
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         �
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*&
_output_shapes
:*

begin_mask*
end_maskM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatConcatV2strided_slice:output:0inputsconcat/axis:output:0*
N*
T0*&
_output_shapes
:�
AssignVariableOpAssignVariableOp(readvariableop_streaming_stream_6_statesconcat:output:0^ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(q
flatten/ConstConst^AssignVariableOp*
_output_shapes
:*
dtype0*
valueB"����0   l
flatten/ReshapeReshapeconcat:output:0flatten/Const:output:0*
T0*
_output_shapes

:0^
IdentityIdentityflatten/Reshape:output:0^NoOp*
T0*
_output_shapes

:0j
NoOpNoOp^AssignVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*'
_input_shapes
:: 2$
AssignVariableOpAssignVariableOp2 
ReadVariableOpReadVariableOp:N J
&
_output_shapes
:
 
_user_specified_nameinputs
�
�
C__inference_stream_4_layer_call_and_return_conditional_losses_10852

inputsI
/concat_readvariableop_streaming_stream_4_states:_
Edepthwise_conv2d_2_depthwise_readvariableop_stream_4_depthwise_kernel:E
7depthwise_conv2d_2_biasadd_readvariableop_stream_4_bias:
identity��AssignVariableOp�concat/ReadVariableOp�)depthwise_conv2d_2/BiasAdd/ReadVariableOp�+depthwise_conv2d_2/depthwise/ReadVariableOp�
concat/ReadVariableOpReadVariableOp/concat_readvariableop_streaming_stream_4_states*&
_output_shapes
:*
dtype0M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatConcatV2concat/ReadVariableOp:value:0inputsconcat/axis:output:0*
N*
T0*&
_output_shapes
:	h
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ����    j
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            j
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         �
strided_sliceStridedSliceconcat:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*&
_output_shapes
:*

begin_mask*
end_mask�
AssignVariableOpAssignVariableOp/concat_readvariableop_streaming_stream_4_statesstrided_slice:output:0^concat/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
+depthwise_conv2d_2/depthwise/ReadVariableOpReadVariableOpEdepthwise_conv2d_2_depthwise_readvariableop_stream_4_depthwise_kernel^AssignVariableOp*&
_output_shapes
:*
dtype0�
"depthwise_conv2d_2/depthwise/ShapeConst^AssignVariableOp*
_output_shapes
:*
dtype0*%
valueB"            �
*depthwise_conv2d_2/depthwise/dilation_rateConst^AssignVariableOp*
_output_shapes
:*
dtype0*
valueB"      �
depthwise_conv2d_2/depthwiseDepthwiseConv2dNativeconcat:output:03depthwise_conv2d_2/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
:*
paddingVALID*
strides
�
)depthwise_conv2d_2/BiasAdd/ReadVariableOpReadVariableOp7depthwise_conv2d_2_biasadd_readvariableop_stream_4_bias^AssignVariableOp*
_output_shapes
:*
dtype0�
depthwise_conv2d_2/BiasAddBiasAdd%depthwise_conv2d_2/depthwise:output:01depthwise_conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:q
IdentityIdentity#depthwise_conv2d_2/BiasAdd:output:0^NoOp*
T0*&
_output_shapes
:�
NoOpNoOp^AssignVariableOp^concat/ReadVariableOp*^depthwise_conv2d_2/BiasAdd/ReadVariableOp,^depthwise_conv2d_2/depthwise/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:: : : 2$
AssignVariableOpAssignVariableOp2.
concat/ReadVariableOpconcat/ReadVariableOp2V
)depthwise_conv2d_2/BiasAdd/ReadVariableOp)depthwise_conv2d_2/BiasAdd/ReadVariableOp2Z
+depthwise_conv2d_2/depthwise/ReadVariableOp+depthwise_conv2d_2/depthwise/ReadVariableOp:N J
&
_output_shapes
:
 
_user_specified_nameinputs
�u
�
@__inference_model_layer_call_and_return_conditional_losses_11220

inputs8
stream_streaming_stream_states:.
stream_stream_kernel: 
stream_stream_bias:E
7batch_normalization_streaming_batch_normalization_gamma:D
6batch_normalization_streaming_batch_normalization_beta:K
=batch_normalization_streaming_batch_normalization_moving_mean:O
Abatch_normalization_streaming_batch_normalization_moving_variance:<
"stream_1_streaming_stream_1_states:<
"stream_1_stream_1_depthwise_kernel:$
stream_1_stream_1_bias:I
;batch_normalization_1_streaming_batch_normalization_1_gamma:H
:batch_normalization_1_streaming_batch_normalization_1_beta:O
Abatch_normalization_1_streaming_batch_normalization_1_moving_mean:S
Ebatch_normalization_1_streaming_batch_normalization_1_moving_variance:<
"stream_2_streaming_stream_2_states:<
"stream_2_stream_2_depthwise_kernel:$
stream_2_stream_2_bias:I
;batch_normalization_2_streaming_batch_normalization_2_gamma:H
:batch_normalization_2_streaming_batch_normalization_2_beta:O
Abatch_normalization_2_streaming_batch_normalization_2_moving_mean:S
Ebatch_normalization_2_streaming_batch_normalization_2_moving_variance:<
"stream_3_streaming_stream_3_states:<
"stream_4_streaming_stream_4_states:<
"stream_4_stream_4_depthwise_kernel:$
stream_4_stream_4_bias:I
;batch_normalization_3_streaming_batch_normalization_3_gamma:H
:batch_normalization_3_streaming_batch_normalization_3_beta:O
Abatch_normalization_3_streaming_batch_normalization_3_moving_mean:S
Ebatch_normalization_3_streaming_batch_normalization_3_moving_variance:<
"stream_5_streaming_stream_5_states:<
"stream_5_stream_5_depthwise_kernel:$
stream_5_stream_5_bias:I
;batch_normalization_4_streaming_batch_normalization_4_gamma:H
:batch_normalization_4_streaming_batch_normalization_4_beta:O
Abatch_normalization_4_streaming_batch_normalization_4_moving_mean:S
Ebatch_normalization_4_streaming_batch_normalization_4_moving_variance:<
"stream_6_streaming_stream_6_states:.
dense_streaming_dense_kernel:0
(
dense_streaming_dense_bias:
2
 dense_1_streaming_dense_1_kernel:
,
dense_1_streaming_dense_1_bias:
identity��+batch_normalization/StatefulPartitionedCall�-batch_normalization_1/StatefulPartitionedCall�-batch_normalization_2/StatefulPartitionedCall�-batch_normalization_3/StatefulPartitionedCall�-batch_normalization_4/StatefulPartitionedCall�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�stream/StatefulPartitionedCall� stream_1/StatefulPartitionedCall� stream_2/StatefulPartitionedCall� stream_3/StatefulPartitionedCall� stream_4/StatefulPartitionedCall� stream_5/StatefulPartitionedCall� stream_6/StatefulPartitionedCall�
max_pooling2d/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:H* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_10222�
stream/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0stream_streaming_stream_statesstream_stream_kernelstream_stream_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_stream_layer_call_and_return_conditional_losses_10725�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'stream/StatefulPartitionedCall:output:07batch_normalization_streaming_batch_normalization_gamma6batch_normalization_streaming_batch_normalization_beta=batch_normalization_streaming_batch_normalization_moving_meanAbatch_normalization_streaming_batch_normalization_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:$*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_10310�
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:$* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_10740�
 stream_1/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0"stream_1_streaming_stream_1_states"stream_1_stream_1_depthwise_kernelstream_1_stream_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_stream_1_layer_call_and_return_conditional_losses_10762�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)stream_1/StatefulPartitionedCall:output:0;batch_normalization_1_streaming_batch_normalization_1_gamma:batch_normalization_1_streaming_batch_normalization_1_betaAbatch_normalization_1_streaming_batch_normalization_1_moving_meanEbatch_normalization_1_streaming_batch_normalization_1_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_10402�
activation_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_10777�
 stream_2/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0"stream_2_streaming_stream_2_states"stream_2_stream_2_depthwise_kernelstream_2_stream_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_stream_2_layer_call_and_return_conditional_losses_10799�
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)stream_2/StatefulPartitionedCall:output:0;batch_normalization_2_streaming_batch_normalization_2_gamma:batch_normalization_2_streaming_batch_normalization_2_betaAbatch_normalization_2_streaming_batch_normalization_2_moving_meanEbatch_normalization_2_streaming_batch_normalization_2_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_10494�
activation_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_activation_2_layer_call_and_return_conditional_losses_10814�
 stream_3/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:0"stream_3_streaming_stream_3_states*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_stream_3_layer_call_and_return_conditional_losses_10829�
 stream_4/StatefulPartitionedCallStatefulPartitionedCall)stream_3/StatefulPartitionedCall:output:0"stream_4_streaming_stream_4_states"stream_4_stream_4_depthwise_kernelstream_4_stream_4_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_stream_4_layer_call_and_return_conditional_losses_10852�
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)stream_4/StatefulPartitionedCall:output:0;batch_normalization_3_streaming_batch_normalization_3_gamma:batch_normalization_3_streaming_batch_normalization_3_betaAbatch_normalization_3_streaming_batch_normalization_3_moving_meanEbatch_normalization_3_streaming_batch_normalization_3_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_10603�
activation_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_activation_3_layer_call_and_return_conditional_losses_10867�
 stream_5/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0"stream_5_streaming_stream_5_states"stream_5_stream_5_depthwise_kernelstream_5_stream_5_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_stream_5_layer_call_and_return_conditional_losses_10889�
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall)stream_5/StatefulPartitionedCall:output:0;batch_normalization_4_streaming_batch_normalization_4_gamma:batch_normalization_4_streaming_batch_normalization_4_betaAbatch_normalization_4_streaming_batch_normalization_4_moving_meanEbatch_normalization_4_streaming_batch_normalization_4_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_10695�
activation_4/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_activation_4_layer_call_and_return_conditional_losses_10904�
dropout/PartitionedCallPartitionedCall%activation_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_11028�
 stream_6/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0"stream_6_streaming_stream_6_states*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_stream_6_layer_call_and_return_conditional_losses_10934�
dense/StatefulPartitionedCallStatefulPartitionedCall)stream_6/StatefulPartitionedCall:output:0dense_streaming_dense_kerneldense_streaming_dense_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_10948�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0 dense_1_streaming_dense_1_kerneldense_1_streaming_dense_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_10963n
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:�
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall^stream/StatefulPartitionedCall!^stream_1/StatefulPartitionedCall!^stream_2/StatefulPartitionedCall!^stream_3/StatefulPartitionedCall!^stream_4/StatefulPartitionedCall!^stream_5/StatefulPartitionedCall!^stream_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*x
_input_shapesg
e:�: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2@
stream/StatefulPartitionedCallstream/StatefulPartitionedCall2D
 stream_1/StatefulPartitionedCall stream_1/StatefulPartitionedCall2D
 stream_2/StatefulPartitionedCall stream_2/StatefulPartitionedCall2D
 stream_3/StatefulPartitionedCall stream_3/StatefulPartitionedCall2D
 stream_4/StatefulPartitionedCall stream_4/StatefulPartitionedCall2D
 stream_5/StatefulPartitionedCall stream_5/StatefulPartitionedCall2D
 stream_6/StatefulPartitionedCall stream_6/StatefulPartitionedCall:O K
'
_output_shapes
:�
 
_user_specified_nameinputs
�"
�
%__inference_model_layer_call_fn_11264

iegm_input1
streaming_stream_states:'
stream_kernel:
stream_bias:1
#streaming_batch_normalization_gamma:0
"streaming_batch_normalization_beta:7
)streaming_batch_normalization_moving_mean:;
-streaming_batch_normalization_moving_variance:3
streaming_stream_1_states:3
stream_1_depthwise_kernel:
stream_1_bias:3
%streaming_batch_normalization_1_gamma:2
$streaming_batch_normalization_1_beta:9
+streaming_batch_normalization_1_moving_mean:=
/streaming_batch_normalization_1_moving_variance:3
streaming_stream_2_states:3
stream_2_depthwise_kernel:
stream_2_bias:3
%streaming_batch_normalization_2_gamma:2
$streaming_batch_normalization_2_beta:9
+streaming_batch_normalization_2_moving_mean:=
/streaming_batch_normalization_2_moving_variance:3
streaming_stream_3_states:3
streaming_stream_4_states:3
stream_4_depthwise_kernel:
stream_4_bias:3
%streaming_batch_normalization_3_gamma:2
$streaming_batch_normalization_3_beta:9
+streaming_batch_normalization_3_moving_mean:=
/streaming_batch_normalization_3_moving_variance:3
streaming_stream_5_states:3
stream_5_depthwise_kernel:
stream_5_bias:3
%streaming_batch_normalization_4_gamma:2
$streaming_batch_normalization_4_beta:9
+streaming_batch_normalization_4_moving_mean:=
/streaming_batch_normalization_4_moving_variance:3
streaming_stream_6_states:(
streaming_dense_kernel:0
"
streaming_dense_bias:
*
streaming_dense_1_kernel:
$
streaming_dense_1_bias:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCall
iegm_inputstreaming_stream_statesstream_kernelstream_bias#streaming_batch_normalization_gamma"streaming_batch_normalization_beta)streaming_batch_normalization_moving_mean-streaming_batch_normalization_moving_variancestreaming_stream_1_statesstream_1_depthwise_kernelstream_1_bias%streaming_batch_normalization_1_gamma$streaming_batch_normalization_1_beta+streaming_batch_normalization_1_moving_mean/streaming_batch_normalization_1_moving_variancestreaming_stream_2_statesstream_2_depthwise_kernelstream_2_bias%streaming_batch_normalization_2_gamma$streaming_batch_normalization_2_beta+streaming_batch_normalization_2_moving_mean/streaming_batch_normalization_2_moving_variancestreaming_stream_3_statesstreaming_stream_4_statesstream_4_depthwise_kernelstream_4_bias%streaming_batch_normalization_3_gamma$streaming_batch_normalization_3_beta+streaming_batch_normalization_3_moving_mean/streaming_batch_normalization_3_moving_variancestreaming_stream_5_statesstream_5_depthwise_kernelstream_5_bias%streaming_batch_normalization_4_gamma$streaming_batch_normalization_4_beta+streaming_batch_normalization_4_moving_mean/streaming_batch_normalization_4_moving_variancestreaming_stream_6_statesstreaming_dense_kernelstreaming_dense_biasstreaming_dense_1_kernelstreaming_dense_1_bias*5
Tin.
,2**
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*D
_read_only_resource_inputs&
$"	
 !"#$&'()*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_11220f
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*x
_input_shapesg
e:�: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
'
_output_shapes
:�
$
_user_specified_name
IEGM_input
�
�
(__inference_stream_3_layer_call_fn_12614

inputs3
streaming_stream_3_states:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstreaming_stream_3_states*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_stream_3_layer_call_and_return_conditional_losses_10829n
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*&
_output_shapes
:`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*'
_input_shapes
:: 22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
:
 
_user_specified_nameinputs
�
�
(__inference_stream_6_layer_call_fn_12844

inputs3
streaming_stream_6_states:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstreaming_stream_6_states*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_stream_6_layer_call_and_return_conditional_losses_10934f
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:0`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*'
_input_shapes
:: 22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
:
 
_user_specified_nameinputs
�

�
5__inference_batch_normalization_1_layer_call_fn_12470

inputs3
%streaming_batch_normalization_1_gamma:2
$streaming_batch_normalization_1_beta:9
+streaming_batch_normalization_1_moving_mean:=
/streaming_batch_normalization_1_moving_variance:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs%streaming_batch_normalization_1_gamma$streaming_batch_normalization_1_beta+streaming_batch_normalization_1_moving_mean/streaming_batch_normalization_1_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_10402�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
%__inference_dense_layer_call_fn_12865

inputs(
streaming_dense_kernel:0
"
streaming_dense_bias:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstreaming_dense_kernelstreaming_dense_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_10948f
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*!
_input_shapes
:0: : 22
StatefulPartitionedCallStatefulPartitionedCall:F B

_output_shapes

:0
 
_user_specified_nameinputs
�
c
G__inference_activation_3_layer_call_and_return_conditional_losses_12719

inputs
identityE
ReluReluinputs*
T0*&
_output_shapes
:Y
IdentityIdentityRelu:activations:0*
T0*&
_output_shapes
:"
identityIdentity:output:0*%
_input_shapes
::N J
&
_output_shapes
:
 
_user_specified_nameinputs
�
C
'__inference_dropout_layer_call_fn_12821

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_11028_
IdentityIdentityPartitionedCall:output:0*
T0*&
_output_shapes
:"
identityIdentity:output:0*%
_input_shapes
::N J
&
_output_shapes
:
 
_user_specified_nameinputs
�
c
G__inference_activation_1_layer_call_and_return_conditional_losses_10777

inputs
identityE
ReluReluinputs*
T0*&
_output_shapes
:Y
IdentityIdentityRelu:activations:0*
T0*&
_output_shapes
:"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
::N J
&
_output_shapes
:
 
_user_specified_nameinputs
�
�
C__inference_stream_1_layer_call_and_return_conditional_losses_10762

inputsI
/concat_readvariableop_streaming_stream_1_states:]
Cdepthwise_conv2d_depthwise_readvariableop_stream_1_depthwise_kernel:C
5depthwise_conv2d_biasadd_readvariableop_stream_1_bias:
identity��AssignVariableOp�concat/ReadVariableOp�'depthwise_conv2d/BiasAdd/ReadVariableOp�)depthwise_conv2d/depthwise/ReadVariableOp�
concat/ReadVariableOpReadVariableOp/concat_readvariableop_streaming_stream_1_states*&
_output_shapes
:*
dtype0M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatConcatV2concat/ReadVariableOp:value:0inputsconcat/axis:output:0*
N*
T0*&
_output_shapes
:'h
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ����    j
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            j
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         �
strided_sliceStridedSliceconcat:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*&
_output_shapes
:*

begin_mask*
end_mask�
AssignVariableOpAssignVariableOp/concat_readvariableop_streaming_stream_1_statesstrided_slice:output:0^concat/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
)depthwise_conv2d/depthwise/ReadVariableOpReadVariableOpCdepthwise_conv2d_depthwise_readvariableop_stream_1_depthwise_kernel^AssignVariableOp*&
_output_shapes
:*
dtype0�
 depthwise_conv2d/depthwise/ShapeConst^AssignVariableOp*
_output_shapes
:*
dtype0*%
valueB"            �
(depthwise_conv2d/depthwise/dilation_rateConst^AssignVariableOp*
_output_shapes
:*
dtype0*
valueB"      �
depthwise_conv2d/depthwiseDepthwiseConv2dNativeconcat:output:01depthwise_conv2d/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
:*
paddingVALID*
strides
�
'depthwise_conv2d/BiasAdd/ReadVariableOpReadVariableOp5depthwise_conv2d_biasadd_readvariableop_stream_1_bias^AssignVariableOp*
_output_shapes
:*
dtype0�
depthwise_conv2d/BiasAddBiasAdd#depthwise_conv2d/depthwise:output:0/depthwise_conv2d/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:o
IdentityIdentity!depthwise_conv2d/BiasAdd:output:0^NoOp*
T0*&
_output_shapes
:�
NoOpNoOp^AssignVariableOp^concat/ReadVariableOp(^depthwise_conv2d/BiasAdd/ReadVariableOp*^depthwise_conv2d/depthwise/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:$: : : 2$
AssignVariableOpAssignVariableOp2.
concat/ReadVariableOpconcat/ReadVariableOp2R
'depthwise_conv2d/BiasAdd/ReadVariableOp'depthwise_conv2d/BiasAdd/ReadVariableOp2V
)depthwise_conv2d/depthwise/ReadVariableOp)depthwise_conv2d/depthwise/ReadVariableOp:N J
&
_output_shapes
:$
 
_user_specified_nameinputs
�
�
A__inference_stream_layer_call_and_return_conditional_losses_12360

inputsG
-concat_readvariableop_streaming_stream_states:D
*conv2d_conv2d_readvariableop_stream_kernel:7
)conv2d_biasadd_readvariableop_stream_bias:
identity��AssignVariableOp�concat/ReadVariableOp�conv2d/BiasAdd/ReadVariableOp�conv2d/Conv2D/ReadVariableOp�
concat/ReadVariableOpReadVariableOp-concat_readvariableop_streaming_stream_states*&
_output_shapes
:*
dtype0M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatConcatV2concat/ReadVariableOp:value:0inputsconcat/axis:output:0*
N*
T0*&
_output_shapes
:Lh
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ����    j
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            j
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         �
strided_sliceStridedSliceconcat:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*&
_output_shapes
:*

begin_mask*
end_mask�
AssignVariableOpAssignVariableOp-concat_readvariableop_streaming_stream_statesstrided_slice:output:0^concat/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
conv2d/Conv2D/ReadVariableOpReadVariableOp*conv2d_conv2d_readvariableop_stream_kernel^AssignVariableOp*&
_output_shapes
:*
dtype0�
conv2d/Conv2DConv2Dconcat:output:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:$*
paddingVALID*
strides
�
conv2d/BiasAdd/ReadVariableOpReadVariableOp)conv2d_biasadd_readvariableop_stream_bias^AssignVariableOp*
_output_shapes
:*
dtype0�
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:$e
IdentityIdentityconv2d/BiasAdd:output:0^NoOp*
T0*&
_output_shapes
:$�
NoOpNoOp^AssignVariableOp^concat/ReadVariableOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*+
_input_shapes
:H: : : 2$
AssignVariableOpAssignVariableOp2.
concat/ReadVariableOpconcat/ReadVariableOp2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp:N J
&
_output_shapes
:H
 
_user_specified_nameinputs
�!
�
%__inference_model_layer_call_fn_11933

inputs1
streaming_stream_states:'
stream_kernel:
stream_bias:1
#streaming_batch_normalization_gamma:0
"streaming_batch_normalization_beta:7
)streaming_batch_normalization_moving_mean:;
-streaming_batch_normalization_moving_variance:3
streaming_stream_1_states:3
stream_1_depthwise_kernel:
stream_1_bias:3
%streaming_batch_normalization_1_gamma:2
$streaming_batch_normalization_1_beta:9
+streaming_batch_normalization_1_moving_mean:=
/streaming_batch_normalization_1_moving_variance:3
streaming_stream_2_states:3
stream_2_depthwise_kernel:
stream_2_bias:3
%streaming_batch_normalization_2_gamma:2
$streaming_batch_normalization_2_beta:9
+streaming_batch_normalization_2_moving_mean:=
/streaming_batch_normalization_2_moving_variance:3
streaming_stream_3_states:3
streaming_stream_4_states:3
stream_4_depthwise_kernel:
stream_4_bias:3
%streaming_batch_normalization_3_gamma:2
$streaming_batch_normalization_3_beta:9
+streaming_batch_normalization_3_moving_mean:=
/streaming_batch_normalization_3_moving_variance:3
streaming_stream_5_states:3
stream_5_depthwise_kernel:
stream_5_bias:3
%streaming_batch_normalization_4_gamma:2
$streaming_batch_normalization_4_beta:9
+streaming_batch_normalization_4_moving_mean:=
/streaming_batch_normalization_4_moving_variance:3
streaming_stream_6_states:(
streaming_dense_kernel:0
"
streaming_dense_bias:
*
streaming_dense_1_kernel:
$
streaming_dense_1_bias:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstreaming_stream_statesstream_kernelstream_bias#streaming_batch_normalization_gamma"streaming_batch_normalization_beta)streaming_batch_normalization_moving_mean-streaming_batch_normalization_moving_variancestreaming_stream_1_statesstream_1_depthwise_kernelstream_1_bias%streaming_batch_normalization_1_gamma$streaming_batch_normalization_1_beta+streaming_batch_normalization_1_moving_mean/streaming_batch_normalization_1_moving_variancestreaming_stream_2_statesstream_2_depthwise_kernelstream_2_bias%streaming_batch_normalization_2_gamma$streaming_batch_normalization_2_beta+streaming_batch_normalization_2_moving_mean/streaming_batch_normalization_2_moving_variancestreaming_stream_3_statesstreaming_stream_4_statesstream_4_depthwise_kernelstream_4_bias%streaming_batch_normalization_3_gamma$streaming_batch_normalization_3_beta+streaming_batch_normalization_3_moving_mean/streaming_batch_normalization_3_moving_variancestreaming_stream_5_statesstream_5_depthwise_kernelstream_5_bias%streaming_batch_normalization_4_gamma$streaming_batch_normalization_4_beta+streaming_batch_normalization_4_moving_mean/streaming_batch_normalization_4_moving_variancestreaming_stream_6_statesstreaming_dense_kernelstreaming_dense_biasstreaming_dense_1_kernelstreaming_dense_1_bias*5
Tin.
,2**
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*D
_read_only_resource_inputs&
$"	
 !"#$&'()*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_11220f
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*x
_input_shapesg
e:�: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:�
 
_user_specified_nameinputs
�	
�
@__inference_dense_layer_call_and_return_conditional_losses_12876

inputs>
,matmul_readvariableop_streaming_dense_kernel:0
9
+biasadd_readvariableop_streaming_dense_bias:

identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOp,matmul_readvariableop_streaming_dense_kernel*
_output_shapes

:0
*
dtype0`
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:
~
BiasAdd/ReadVariableOpReadVariableOp+biasadd_readvariableop_streaming_dense_bias*
_output_shapes
:
*
dtype0m
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:
G
ReluReluBiasAdd:output:0*
T0*
_output_shapes

:
X
IdentityIdentityRelu:activations:0^NoOp*
T0*
_output_shapes

:
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*!
_input_shapes
:0: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:F B

_output_shapes

:0
 
_user_specified_nameinputs
�
�
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_12691

inputsB
4readvariableop_streaming_batch_normalization_3_gamma:C
5readvariableop_1_streaming_batch_normalization_3_beta:Y
Kfusedbatchnormv3_readvariableop_streaming_batch_normalization_3_moving_mean:_
Qfusedbatchnormv3_readvariableop_1_streaming_batch_normalization_3_moving_variance:
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1
ReadVariableOpReadVariableOp4readvariableop_streaming_batch_normalization_3_gamma*
_output_shapes
:*
dtype0�
ReadVariableOp_1ReadVariableOp5readvariableop_1_streaming_batch_normalization_3_beta*
_output_shapes
:*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOpKfusedbatchnormv3_readvariableop_streaming_batch_normalization_3_moving_mean*
_output_shapes
:*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpQfusedbatchnormv3_readvariableop_1_streaming_batch_normalization_3_moving_variance*
_output_shapes
:*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%��'7*
exponential_avg_factor%fff?�
AssignNewValueAssignVariableOpKfusedbatchnormv3_readvariableop_streaming_batch_normalization_3_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOpQfusedbatchnormv3_readvariableop_1_streaming_batch_normalization_3_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
c
G__inference_activation_3_layer_call_and_return_conditional_losses_10867

inputs
identityE
ReluReluinputs*
T0*&
_output_shapes
:Y
IdentityIdentityRelu:activations:0*
T0*&
_output_shapes
:"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
::N J
&
_output_shapes
:
 
_user_specified_nameinputs
�
�
&__inference_stream_layer_call_fn_12342

inputs1
streaming_stream_states:'
stream_kernel:
stream_bias:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstreaming_stream_statesstream_kernelstream_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_stream_layer_call_and_return_conditional_losses_10725n
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*&
_output_shapes
:$`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*+
_input_shapes
:H: : : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
:H
 
_user_specified_nameinputs
��
�1
@__inference_model_layer_call_and_return_conditional_losses_12132

inputsN
4stream_concat_readvariableop_streaming_stream_states:K
1stream_conv2d_conv2d_readvariableop_stream_kernel:>
0stream_conv2d_biasadd_readvariableop_stream_bias:T
Fbatch_normalization_readvariableop_streaming_batch_normalization_gamma:U
Gbatch_normalization_readvariableop_1_streaming_batch_normalization_beta:k
]batch_normalization_fusedbatchnormv3_readvariableop_streaming_batch_normalization_moving_mean:q
cbatch_normalization_fusedbatchnormv3_readvariableop_1_streaming_batch_normalization_moving_variance:R
8stream_1_concat_readvariableop_streaming_stream_1_states:f
Lstream_1_depthwise_conv2d_depthwise_readvariableop_stream_1_depthwise_kernel:L
>stream_1_depthwise_conv2d_biasadd_readvariableop_stream_1_bias:X
Jbatch_normalization_1_readvariableop_streaming_batch_normalization_1_gamma:Y
Kbatch_normalization_1_readvariableop_1_streaming_batch_normalization_1_beta:o
abatch_normalization_1_fusedbatchnormv3_readvariableop_streaming_batch_normalization_1_moving_mean:u
gbatch_normalization_1_fusedbatchnormv3_readvariableop_1_streaming_batch_normalization_1_moving_variance:R
8stream_2_concat_readvariableop_streaming_stream_2_states:h
Nstream_2_depthwise_conv2d_1_depthwise_readvariableop_stream_2_depthwise_kernel:N
@stream_2_depthwise_conv2d_1_biasadd_readvariableop_stream_2_bias:X
Jbatch_normalization_2_readvariableop_streaming_batch_normalization_2_gamma:Y
Kbatch_normalization_2_readvariableop_1_streaming_batch_normalization_2_beta:o
abatch_normalization_2_fusedbatchnormv3_readvariableop_streaming_batch_normalization_2_moving_mean:u
gbatch_normalization_2_fusedbatchnormv3_readvariableop_1_streaming_batch_normalization_2_moving_variance:R
8stream_3_concat_readvariableop_streaming_stream_3_states:R
8stream_4_concat_readvariableop_streaming_stream_4_states:h
Nstream_4_depthwise_conv2d_2_depthwise_readvariableop_stream_4_depthwise_kernel:N
@stream_4_depthwise_conv2d_2_biasadd_readvariableop_stream_4_bias:X
Jbatch_normalization_3_readvariableop_streaming_batch_normalization_3_gamma:Y
Kbatch_normalization_3_readvariableop_1_streaming_batch_normalization_3_beta:o
abatch_normalization_3_fusedbatchnormv3_readvariableop_streaming_batch_normalization_3_moving_mean:u
gbatch_normalization_3_fusedbatchnormv3_readvariableop_1_streaming_batch_normalization_3_moving_variance:R
8stream_5_concat_readvariableop_streaming_stream_5_states:h
Nstream_5_depthwise_conv2d_3_depthwise_readvariableop_stream_5_depthwise_kernel:N
@stream_5_depthwise_conv2d_3_biasadd_readvariableop_stream_5_bias:X
Jbatch_normalization_4_readvariableop_streaming_batch_normalization_4_gamma:Y
Kbatch_normalization_4_readvariableop_1_streaming_batch_normalization_4_beta:o
abatch_normalization_4_fusedbatchnormv3_readvariableop_streaming_batch_normalization_4_moving_mean:u
gbatch_normalization_4_fusedbatchnormv3_readvariableop_1_streaming_batch_normalization_4_moving_variance:K
1stream_6_readvariableop_streaming_stream_6_states:D
2dense_matmul_readvariableop_streaming_dense_kernel:0
?
1dense_biasadd_readvariableop_streaming_dense_bias:
H
6dense_1_matmul_readvariableop_streaming_dense_1_kernel:
C
5dense_1_biasadd_readvariableop_streaming_dense_1_bias:
identity��"batch_normalization/AssignNewValue�$batch_normalization/AssignNewValue_1�3batch_normalization/FusedBatchNormV3/ReadVariableOp�5batch_normalization/FusedBatchNormV3/ReadVariableOp_1�"batch_normalization/ReadVariableOp�$batch_normalization/ReadVariableOp_1�$batch_normalization_1/AssignNewValue�&batch_normalization_1/AssignNewValue_1�5batch_normalization_1/FusedBatchNormV3/ReadVariableOp�7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_1/ReadVariableOp�&batch_normalization_1/ReadVariableOp_1�$batch_normalization_2/AssignNewValue�&batch_normalization_2/AssignNewValue_1�5batch_normalization_2/FusedBatchNormV3/ReadVariableOp�7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_2/ReadVariableOp�&batch_normalization_2/ReadVariableOp_1�$batch_normalization_3/AssignNewValue�&batch_normalization_3/AssignNewValue_1�5batch_normalization_3/FusedBatchNormV3/ReadVariableOp�7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_3/ReadVariableOp�&batch_normalization_3/ReadVariableOp_1�$batch_normalization_4/AssignNewValue�&batch_normalization_4/AssignNewValue_1�5batch_normalization_4/FusedBatchNormV3/ReadVariableOp�7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_4/ReadVariableOp�&batch_normalization_4/ReadVariableOp_1�dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�stream/AssignVariableOp�stream/concat/ReadVariableOp�$stream/conv2d/BiasAdd/ReadVariableOp�#stream/conv2d/Conv2D/ReadVariableOp�stream_1/AssignVariableOp�stream_1/concat/ReadVariableOp�0stream_1/depthwise_conv2d/BiasAdd/ReadVariableOp�2stream_1/depthwise_conv2d/depthwise/ReadVariableOp�stream_2/AssignVariableOp�stream_2/concat/ReadVariableOp�2stream_2/depthwise_conv2d_1/BiasAdd/ReadVariableOp�4stream_2/depthwise_conv2d_1/depthwise/ReadVariableOp�stream_3/AssignVariableOp�stream_3/concat/ReadVariableOp�stream_4/AssignVariableOp�stream_4/concat/ReadVariableOp�2stream_4/depthwise_conv2d_2/BiasAdd/ReadVariableOp�4stream_4/depthwise_conv2d_2/depthwise/ReadVariableOp�stream_5/AssignVariableOp�stream_5/concat/ReadVariableOp�2stream_5/depthwise_conv2d_3/BiasAdd/ReadVariableOp�4stream_5/depthwise_conv2d_3/depthwise/ReadVariableOp�stream_6/AssignVariableOp�stream_6/ReadVariableOp�
max_pooling2d/MaxPoolMaxPoolinputs*&
_output_shapes
:H*
ksize
*
paddingVALID*
strides
�
stream/concat/ReadVariableOpReadVariableOp4stream_concat_readvariableop_streaming_stream_states*&
_output_shapes
:*
dtype0T
stream/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
stream/concatConcatV2$stream/concat/ReadVariableOp:value:0max_pooling2d/MaxPool:output:0stream/concat/axis:output:0*
N*
T0*&
_output_shapes
:Lo
stream/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ����    q
stream/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            q
stream/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         �
stream/strided_sliceStridedSlicestream/concat:output:0#stream/strided_slice/stack:output:0%stream/strided_slice/stack_1:output:0%stream/strided_slice/stack_2:output:0*
Index0*
T0*&
_output_shapes
:*

begin_mask*
end_mask�
stream/AssignVariableOpAssignVariableOp4stream_concat_readvariableop_streaming_stream_statesstream/strided_slice:output:0^stream/concat/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
#stream/conv2d/Conv2D/ReadVariableOpReadVariableOp1stream_conv2d_conv2d_readvariableop_stream_kernel^stream/AssignVariableOp*&
_output_shapes
:*
dtype0�
stream/conv2d/Conv2DConv2Dstream/concat:output:0+stream/conv2d/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:$*
paddingVALID*
strides
�
$stream/conv2d/BiasAdd/ReadVariableOpReadVariableOp0stream_conv2d_biasadd_readvariableop_stream_bias^stream/AssignVariableOp*
_output_shapes
:*
dtype0�
stream/conv2d/BiasAddBiasAddstream/conv2d/Conv2D:output:0,stream/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:$�
"batch_normalization/ReadVariableOpReadVariableOpFbatch_normalization_readvariableop_streaming_batch_normalization_gamma*
_output_shapes
:*
dtype0�
$batch_normalization/ReadVariableOp_1ReadVariableOpGbatch_normalization_readvariableop_1_streaming_batch_normalization_beta*
_output_shapes
:*
dtype0�
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp]batch_normalization_fusedbatchnormv3_readvariableop_streaming_batch_normalization_moving_mean*
_output_shapes
:*
dtype0�
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpcbatch_normalization_fusedbatchnormv3_readvariableop_1_streaming_batch_normalization_moving_variance*
_output_shapes
:*
dtype0�
$batch_normalization/FusedBatchNormV3FusedBatchNormV3stream/conv2d/BiasAdd:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:$:::::*
epsilon%��'7*
exponential_avg_factor%fff?�
"batch_normalization/AssignNewValueAssignVariableOp]batch_normalization_fusedbatchnormv3_readvariableop_streaming_batch_normalization_moving_mean1batch_normalization/FusedBatchNormV3:batch_mean:04^batch_normalization/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
$batch_normalization/AssignNewValue_1AssignVariableOpcbatch_normalization_fusedbatchnormv3_readvariableop_1_streaming_batch_normalization_moving_variance5batch_normalization/FusedBatchNormV3:batch_variance:06^batch_normalization/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(r
activation/ReluRelu(batch_normalization/FusedBatchNormV3:y:0*
T0*&
_output_shapes
:$�
stream_1/concat/ReadVariableOpReadVariableOp8stream_1_concat_readvariableop_streaming_stream_1_states*&
_output_shapes
:*
dtype0V
stream_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
stream_1/concatConcatV2&stream_1/concat/ReadVariableOp:value:0activation/Relu:activations:0stream_1/concat/axis:output:0*
N*
T0*&
_output_shapes
:'q
stream_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ����    s
stream_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            s
stream_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         �
stream_1/strided_sliceStridedSlicestream_1/concat:output:0%stream_1/strided_slice/stack:output:0'stream_1/strided_slice/stack_1:output:0'stream_1/strided_slice/stack_2:output:0*
Index0*
T0*&
_output_shapes
:*

begin_mask*
end_mask�
stream_1/AssignVariableOpAssignVariableOp8stream_1_concat_readvariableop_streaming_stream_1_statesstream_1/strided_slice:output:0^stream_1/concat/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
2stream_1/depthwise_conv2d/depthwise/ReadVariableOpReadVariableOpLstream_1_depthwise_conv2d_depthwise_readvariableop_stream_1_depthwise_kernel^stream_1/AssignVariableOp*&
_output_shapes
:*
dtype0�
)stream_1/depthwise_conv2d/depthwise/ShapeConst^stream_1/AssignVariableOp*
_output_shapes
:*
dtype0*%
valueB"            �
1stream_1/depthwise_conv2d/depthwise/dilation_rateConst^stream_1/AssignVariableOp*
_output_shapes
:*
dtype0*
valueB"      �
#stream_1/depthwise_conv2d/depthwiseDepthwiseConv2dNativestream_1/concat:output:0:stream_1/depthwise_conv2d/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
:*
paddingVALID*
strides
�
0stream_1/depthwise_conv2d/BiasAdd/ReadVariableOpReadVariableOp>stream_1_depthwise_conv2d_biasadd_readvariableop_stream_1_bias^stream_1/AssignVariableOp*
_output_shapes
:*
dtype0�
!stream_1/depthwise_conv2d/BiasAddBiasAdd,stream_1/depthwise_conv2d/depthwise:output:08stream_1/depthwise_conv2d/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:�
$batch_normalization_1/ReadVariableOpReadVariableOpJbatch_normalization_1_readvariableop_streaming_batch_normalization_1_gamma*
_output_shapes
:*
dtype0�
&batch_normalization_1/ReadVariableOp_1ReadVariableOpKbatch_normalization_1_readvariableop_1_streaming_batch_normalization_1_beta*
_output_shapes
:*
dtype0�
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpabatch_normalization_1_fusedbatchnormv3_readvariableop_streaming_batch_normalization_1_moving_mean*
_output_shapes
:*
dtype0�
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpgbatch_normalization_1_fusedbatchnormv3_readvariableop_1_streaming_batch_normalization_1_moving_variance*
_output_shapes
:*
dtype0�
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3*stream_1/depthwise_conv2d/BiasAdd:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.::::::*
epsilon%��'7*
exponential_avg_factor%fff?�
$batch_normalization_1/AssignNewValueAssignVariableOpabatch_normalization_1_fusedbatchnormv3_readvariableop_streaming_batch_normalization_1_moving_mean3batch_normalization_1/FusedBatchNormV3:batch_mean:06^batch_normalization_1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
&batch_normalization_1/AssignNewValue_1AssignVariableOpgbatch_normalization_1_fusedbatchnormv3_readvariableop_1_streaming_batch_normalization_1_moving_variance7batch_normalization_1/FusedBatchNormV3:batch_variance:08^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(v
activation_1/ReluRelu*batch_normalization_1/FusedBatchNormV3:y:0*
T0*&
_output_shapes
:�
stream_2/concat/ReadVariableOpReadVariableOp8stream_2_concat_readvariableop_streaming_stream_2_states*&
_output_shapes
:*
dtype0V
stream_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
stream_2/concatConcatV2&stream_2/concat/ReadVariableOp:value:0activation_1/Relu:activations:0stream_2/concat/axis:output:0*
N*
T0*&
_output_shapes
:q
stream_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ����    s
stream_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            s
stream_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         �
stream_2/strided_sliceStridedSlicestream_2/concat:output:0%stream_2/strided_slice/stack:output:0'stream_2/strided_slice/stack_1:output:0'stream_2/strided_slice/stack_2:output:0*
Index0*
T0*&
_output_shapes
:*

begin_mask*
end_mask�
stream_2/AssignVariableOpAssignVariableOp8stream_2_concat_readvariableop_streaming_stream_2_statesstream_2/strided_slice:output:0^stream_2/concat/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
4stream_2/depthwise_conv2d_1/depthwise/ReadVariableOpReadVariableOpNstream_2_depthwise_conv2d_1_depthwise_readvariableop_stream_2_depthwise_kernel^stream_2/AssignVariableOp*&
_output_shapes
:*
dtype0�
+stream_2/depthwise_conv2d_1/depthwise/ShapeConst^stream_2/AssignVariableOp*
_output_shapes
:*
dtype0*%
valueB"            �
3stream_2/depthwise_conv2d_1/depthwise/dilation_rateConst^stream_2/AssignVariableOp*
_output_shapes
:*
dtype0*
valueB"      �
%stream_2/depthwise_conv2d_1/depthwiseDepthwiseConv2dNativestream_2/concat:output:0<stream_2/depthwise_conv2d_1/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
:*
paddingVALID*
strides
�
2stream_2/depthwise_conv2d_1/BiasAdd/ReadVariableOpReadVariableOp@stream_2_depthwise_conv2d_1_biasadd_readvariableop_stream_2_bias^stream_2/AssignVariableOp*
_output_shapes
:*
dtype0�
#stream_2/depthwise_conv2d_1/BiasAddBiasAdd.stream_2/depthwise_conv2d_1/depthwise:output:0:stream_2/depthwise_conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:�
$batch_normalization_2/ReadVariableOpReadVariableOpJbatch_normalization_2_readvariableop_streaming_batch_normalization_2_gamma*
_output_shapes
:*
dtype0�
&batch_normalization_2/ReadVariableOp_1ReadVariableOpKbatch_normalization_2_readvariableop_1_streaming_batch_normalization_2_beta*
_output_shapes
:*
dtype0�
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpabatch_normalization_2_fusedbatchnormv3_readvariableop_streaming_batch_normalization_2_moving_mean*
_output_shapes
:*
dtype0�
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpgbatch_normalization_2_fusedbatchnormv3_readvariableop_1_streaming_batch_normalization_2_moving_variance*
_output_shapes
:*
dtype0�
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3,stream_2/depthwise_conv2d_1/BiasAdd:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.::::::*
epsilon%��'7*
exponential_avg_factor%fff?�
$batch_normalization_2/AssignNewValueAssignVariableOpabatch_normalization_2_fusedbatchnormv3_readvariableop_streaming_batch_normalization_2_moving_mean3batch_normalization_2/FusedBatchNormV3:batch_mean:06^batch_normalization_2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
&batch_normalization_2/AssignNewValue_1AssignVariableOpgbatch_normalization_2_fusedbatchnormv3_readvariableop_1_streaming_batch_normalization_2_moving_variance7batch_normalization_2/FusedBatchNormV3:batch_variance:08^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(v
activation_2/ReluRelu*batch_normalization_2/FusedBatchNormV3:y:0*
T0*&
_output_shapes
:�
stream_3/concat/ReadVariableOpReadVariableOp8stream_3_concat_readvariableop_streaming_stream_3_states*&
_output_shapes
:*
dtype0V
stream_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
stream_3/concatConcatV2&stream_3/concat/ReadVariableOp:value:0activation_2/Relu:activations:0stream_3/concat/axis:output:0*
N*
T0*&
_output_shapes
:q
stream_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ����    s
stream_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            s
stream_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         �
stream_3/strided_sliceStridedSlicestream_3/concat:output:0%stream_3/strided_slice/stack:output:0'stream_3/strided_slice/stack_1:output:0'stream_3/strided_slice/stack_2:output:0*
Index0*
T0*&
_output_shapes
:*

begin_mask*
end_mask�
stream_3/AssignVariableOpAssignVariableOp8stream_3_concat_readvariableop_streaming_stream_3_statesstream_3/strided_slice:output:0^stream_3/concat/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
"stream_3/average_pooling2d/AvgPoolAvgPoolstream_3/concat:output:0^stream_3/AssignVariableOp*
T0*&
_output_shapes
:*
ksize
*
paddingVALID*
strides
�
stream_4/concat/ReadVariableOpReadVariableOp8stream_4_concat_readvariableop_streaming_stream_4_states*&
_output_shapes
:*
dtype0V
stream_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
stream_4/concatConcatV2&stream_4/concat/ReadVariableOp:value:0+stream_3/average_pooling2d/AvgPool:output:0stream_4/concat/axis:output:0*
N*
T0*&
_output_shapes
:	q
stream_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ����    s
stream_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            s
stream_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         �
stream_4/strided_sliceStridedSlicestream_4/concat:output:0%stream_4/strided_slice/stack:output:0'stream_4/strided_slice/stack_1:output:0'stream_4/strided_slice/stack_2:output:0*
Index0*
T0*&
_output_shapes
:*

begin_mask*
end_mask�
stream_4/AssignVariableOpAssignVariableOp8stream_4_concat_readvariableop_streaming_stream_4_statesstream_4/strided_slice:output:0^stream_4/concat/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
4stream_4/depthwise_conv2d_2/depthwise/ReadVariableOpReadVariableOpNstream_4_depthwise_conv2d_2_depthwise_readvariableop_stream_4_depthwise_kernel^stream_4/AssignVariableOp*&
_output_shapes
:*
dtype0�
+stream_4/depthwise_conv2d_2/depthwise/ShapeConst^stream_4/AssignVariableOp*
_output_shapes
:*
dtype0*%
valueB"            �
3stream_4/depthwise_conv2d_2/depthwise/dilation_rateConst^stream_4/AssignVariableOp*
_output_shapes
:*
dtype0*
valueB"      �
%stream_4/depthwise_conv2d_2/depthwiseDepthwiseConv2dNativestream_4/concat:output:0<stream_4/depthwise_conv2d_2/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
:*
paddingVALID*
strides
�
2stream_4/depthwise_conv2d_2/BiasAdd/ReadVariableOpReadVariableOp@stream_4_depthwise_conv2d_2_biasadd_readvariableop_stream_4_bias^stream_4/AssignVariableOp*
_output_shapes
:*
dtype0�
#stream_4/depthwise_conv2d_2/BiasAddBiasAdd.stream_4/depthwise_conv2d_2/depthwise:output:0:stream_4/depthwise_conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:�
$batch_normalization_3/ReadVariableOpReadVariableOpJbatch_normalization_3_readvariableop_streaming_batch_normalization_3_gamma*
_output_shapes
:*
dtype0�
&batch_normalization_3/ReadVariableOp_1ReadVariableOpKbatch_normalization_3_readvariableop_1_streaming_batch_normalization_3_beta*
_output_shapes
:*
dtype0�
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpabatch_normalization_3_fusedbatchnormv3_readvariableop_streaming_batch_normalization_3_moving_mean*
_output_shapes
:*
dtype0�
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpgbatch_normalization_3_fusedbatchnormv3_readvariableop_1_streaming_batch_normalization_3_moving_variance*
_output_shapes
:*
dtype0�
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3,stream_4/depthwise_conv2d_2/BiasAdd:output:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.::::::*
epsilon%��'7*
exponential_avg_factor%fff?�
$batch_normalization_3/AssignNewValueAssignVariableOpabatch_normalization_3_fusedbatchnormv3_readvariableop_streaming_batch_normalization_3_moving_mean3batch_normalization_3/FusedBatchNormV3:batch_mean:06^batch_normalization_3/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
&batch_normalization_3/AssignNewValue_1AssignVariableOpgbatch_normalization_3_fusedbatchnormv3_readvariableop_1_streaming_batch_normalization_3_moving_variance7batch_normalization_3/FusedBatchNormV3:batch_variance:08^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(v
activation_3/ReluRelu*batch_normalization_3/FusedBatchNormV3:y:0*
T0*&
_output_shapes
:�
stream_5/concat/ReadVariableOpReadVariableOp8stream_5_concat_readvariableop_streaming_stream_5_states*&
_output_shapes
:*
dtype0V
stream_5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
stream_5/concatConcatV2&stream_5/concat/ReadVariableOp:value:0activation_3/Relu:activations:0stream_5/concat/axis:output:0*
N*
T0*&
_output_shapes
:q
stream_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ����    s
stream_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            s
stream_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         �
stream_5/strided_sliceStridedSlicestream_5/concat:output:0%stream_5/strided_slice/stack:output:0'stream_5/strided_slice/stack_1:output:0'stream_5/strided_slice/stack_2:output:0*
Index0*
T0*&
_output_shapes
:*

begin_mask*
end_mask�
stream_5/AssignVariableOpAssignVariableOp8stream_5_concat_readvariableop_streaming_stream_5_statesstream_5/strided_slice:output:0^stream_5/concat/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
4stream_5/depthwise_conv2d_3/depthwise/ReadVariableOpReadVariableOpNstream_5_depthwise_conv2d_3_depthwise_readvariableop_stream_5_depthwise_kernel^stream_5/AssignVariableOp*&
_output_shapes
:*
dtype0�
+stream_5/depthwise_conv2d_3/depthwise/ShapeConst^stream_5/AssignVariableOp*
_output_shapes
:*
dtype0*%
valueB"            �
3stream_5/depthwise_conv2d_3/depthwise/dilation_rateConst^stream_5/AssignVariableOp*
_output_shapes
:*
dtype0*
valueB"      �
%stream_5/depthwise_conv2d_3/depthwiseDepthwiseConv2dNativestream_5/concat:output:0<stream_5/depthwise_conv2d_3/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
:*
paddingVALID*
strides
�
2stream_5/depthwise_conv2d_3/BiasAdd/ReadVariableOpReadVariableOp@stream_5_depthwise_conv2d_3_biasadd_readvariableop_stream_5_bias^stream_5/AssignVariableOp*
_output_shapes
:*
dtype0�
#stream_5/depthwise_conv2d_3/BiasAddBiasAdd.stream_5/depthwise_conv2d_3/depthwise:output:0:stream_5/depthwise_conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:�
$batch_normalization_4/ReadVariableOpReadVariableOpJbatch_normalization_4_readvariableop_streaming_batch_normalization_4_gamma*
_output_shapes
:*
dtype0�
&batch_normalization_4/ReadVariableOp_1ReadVariableOpKbatch_normalization_4_readvariableop_1_streaming_batch_normalization_4_beta*
_output_shapes
:*
dtype0�
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpabatch_normalization_4_fusedbatchnormv3_readvariableop_streaming_batch_normalization_4_moving_mean*
_output_shapes
:*
dtype0�
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpgbatch_normalization_4_fusedbatchnormv3_readvariableop_1_streaming_batch_normalization_4_moving_variance*
_output_shapes
:*
dtype0�
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3,stream_5/depthwise_conv2d_3/BiasAdd:output:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.::::::*
epsilon%��'7*
exponential_avg_factor%fff?�
$batch_normalization_4/AssignNewValueAssignVariableOpabatch_normalization_4_fusedbatchnormv3_readvariableop_streaming_batch_normalization_4_moving_mean3batch_normalization_4/FusedBatchNormV3:batch_mean:06^batch_normalization_4/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
&batch_normalization_4/AssignNewValue_1AssignVariableOpgbatch_normalization_4_fusedbatchnormv3_readvariableop_1_streaming_batch_normalization_4_moving_variance7batch_normalization_4/FusedBatchNormV3:batch_variance:08^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(v
activation_4/ReluRelu*batch_normalization_4/FusedBatchNormV3:y:0*
T0*&
_output_shapes
:Z
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
dropout/dropout/MulMulactivation_4/Relu:activations:0dropout/dropout/Const:output:0*
T0*&
_output_shapes
:n
dropout/dropout/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            �
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*&
_output_shapes
:*
dtype0c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*&
_output_shapes
:\
dropout/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/dropout/SelectV2SelectV2 dropout/dropout/GreaterEqual:z:0dropout/dropout/Mul:z:0 dropout/dropout/Const_1:output:0*
T0*&
_output_shapes
:�
stream_6/ReadVariableOpReadVariableOp1stream_6_readvariableop_streaming_stream_6_states*&
_output_shapes
:*
dtype0q
stream_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           s
stream_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           s
stream_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         �
stream_6/strided_sliceStridedSlicestream_6/ReadVariableOp:value:0%stream_6/strided_slice/stack:output:0'stream_6/strided_slice/stack_1:output:0'stream_6/strided_slice/stack_2:output:0*
Index0*
T0*&
_output_shapes
:*

begin_mask*
end_maskV
stream_6/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
stream_6/concatConcatV2stream_6/strided_slice:output:0!dropout/dropout/SelectV2:output:0stream_6/concat/axis:output:0*
N*
T0*&
_output_shapes
:�
stream_6/AssignVariableOpAssignVariableOp1stream_6_readvariableop_streaming_stream_6_statesstream_6/concat:output:0^stream_6/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
stream_6/flatten/ConstConst^stream_6/AssignVariableOp*
_output_shapes
:*
dtype0*
valueB"����0   �
stream_6/flatten/ReshapeReshapestream_6/concat:output:0stream_6/flatten/Const:output:0*
T0*
_output_shapes

:0�
dense/MatMul/ReadVariableOpReadVariableOp2dense_matmul_readvariableop_streaming_dense_kernel*
_output_shapes

:0
*
dtype0�
dense/MatMulMatMul!stream_6/flatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:
�
dense/BiasAdd/ReadVariableOpReadVariableOp1dense_biasadd_readvariableop_streaming_dense_bias*
_output_shapes
:
*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:
S

dense/ReluReludense/BiasAdd:output:0*
T0*
_output_shapes

:
�
dense_1/MatMul/ReadVariableOpReadVariableOp6dense_1_matmul_readvariableop_streaming_dense_1_kernel*
_output_shapes

:
*
dtype0�
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:�
dense_1/BiasAdd/ReadVariableOpReadVariableOp5dense_1_biasadd_readvariableop_streaming_dense_1_bias*
_output_shapes
:*
dtype0�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:]
dense_1/SoftmaxSoftmaxdense_1/BiasAdd:output:0*
T0*
_output_shapes

:_
IdentityIdentitydense_1/Softmax:softmax:0^NoOp*
T0*
_output_shapes

:�
NoOpNoOp#^batch_normalization/AssignNewValue%^batch_normalization/AssignNewValue_14^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_1%^batch_normalization_1/AssignNewValue'^batch_normalization_1/AssignNewValue_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_1%^batch_normalization_2/AssignNewValue'^batch_normalization_2/AssignNewValue_16^batch_normalization_2/FusedBatchNormV3/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_1%^batch_normalization_3/AssignNewValue'^batch_normalization_3/AssignNewValue_16^batch_normalization_3/FusedBatchNormV3/ReadVariableOp8^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_3/ReadVariableOp'^batch_normalization_3/ReadVariableOp_1%^batch_normalization_4/AssignNewValue'^batch_normalization_4/AssignNewValue_16^batch_normalization_4/FusedBatchNormV3/ReadVariableOp8^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_4/ReadVariableOp'^batch_normalization_4/ReadVariableOp_1^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^stream/AssignVariableOp^stream/concat/ReadVariableOp%^stream/conv2d/BiasAdd/ReadVariableOp$^stream/conv2d/Conv2D/ReadVariableOp^stream_1/AssignVariableOp^stream_1/concat/ReadVariableOp1^stream_1/depthwise_conv2d/BiasAdd/ReadVariableOp3^stream_1/depthwise_conv2d/depthwise/ReadVariableOp^stream_2/AssignVariableOp^stream_2/concat/ReadVariableOp3^stream_2/depthwise_conv2d_1/BiasAdd/ReadVariableOp5^stream_2/depthwise_conv2d_1/depthwise/ReadVariableOp^stream_3/AssignVariableOp^stream_3/concat/ReadVariableOp^stream_4/AssignVariableOp^stream_4/concat/ReadVariableOp3^stream_4/depthwise_conv2d_2/BiasAdd/ReadVariableOp5^stream_4/depthwise_conv2d_2/depthwise/ReadVariableOp^stream_5/AssignVariableOp^stream_5/concat/ReadVariableOp3^stream_5/depthwise_conv2d_3/BiasAdd/ReadVariableOp5^stream_5/depthwise_conv2d_3/depthwise/ReadVariableOp^stream_6/AssignVariableOp^stream_6/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*x
_input_shapesg
e:�: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2L
$batch_normalization/AssignNewValue_1$batch_normalization/AssignNewValue_12H
"batch_normalization/AssignNewValue"batch_normalization/AssignNewValue2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2P
&batch_normalization_1/AssignNewValue_1&batch_normalization_1/AssignNewValue_12L
$batch_normalization_1/AssignNewValue$batch_normalization_1/AssignNewValue2r
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_17batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12n
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_12L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2P
&batch_normalization_2/AssignNewValue_1&batch_normalization_2/AssignNewValue_12L
$batch_normalization_2/AssignNewValue$batch_normalization_2/AssignNewValue2r
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_17batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12n
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp5batch_normalization_2/FusedBatchNormV3/ReadVariableOp2P
&batch_normalization_2/ReadVariableOp_1&batch_normalization_2/ReadVariableOp_12L
$batch_normalization_2/ReadVariableOp$batch_normalization_2/ReadVariableOp2P
&batch_normalization_3/AssignNewValue_1&batch_normalization_3/AssignNewValue_12L
$batch_normalization_3/AssignNewValue$batch_normalization_3/AssignNewValue2r
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_17batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12n
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp5batch_normalization_3/FusedBatchNormV3/ReadVariableOp2P
&batch_normalization_3/ReadVariableOp_1&batch_normalization_3/ReadVariableOp_12L
$batch_normalization_3/ReadVariableOp$batch_normalization_3/ReadVariableOp2P
&batch_normalization_4/AssignNewValue_1&batch_normalization_4/AssignNewValue_12L
$batch_normalization_4/AssignNewValue$batch_normalization_4/AssignNewValue2r
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_17batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12n
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp5batch_normalization_4/FusedBatchNormV3/ReadVariableOp2P
&batch_normalization_4/ReadVariableOp_1&batch_normalization_4/ReadVariableOp_12L
$batch_normalization_4/ReadVariableOp$batch_normalization_4/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp22
stream/AssignVariableOpstream/AssignVariableOp2<
stream/concat/ReadVariableOpstream/concat/ReadVariableOp2L
$stream/conv2d/BiasAdd/ReadVariableOp$stream/conv2d/BiasAdd/ReadVariableOp2J
#stream/conv2d/Conv2D/ReadVariableOp#stream/conv2d/Conv2D/ReadVariableOp26
stream_1/AssignVariableOpstream_1/AssignVariableOp2@
stream_1/concat/ReadVariableOpstream_1/concat/ReadVariableOp2d
0stream_1/depthwise_conv2d/BiasAdd/ReadVariableOp0stream_1/depthwise_conv2d/BiasAdd/ReadVariableOp2h
2stream_1/depthwise_conv2d/depthwise/ReadVariableOp2stream_1/depthwise_conv2d/depthwise/ReadVariableOp26
stream_2/AssignVariableOpstream_2/AssignVariableOp2@
stream_2/concat/ReadVariableOpstream_2/concat/ReadVariableOp2h
2stream_2/depthwise_conv2d_1/BiasAdd/ReadVariableOp2stream_2/depthwise_conv2d_1/BiasAdd/ReadVariableOp2l
4stream_2/depthwise_conv2d_1/depthwise/ReadVariableOp4stream_2/depthwise_conv2d_1/depthwise/ReadVariableOp26
stream_3/AssignVariableOpstream_3/AssignVariableOp2@
stream_3/concat/ReadVariableOpstream_3/concat/ReadVariableOp26
stream_4/AssignVariableOpstream_4/AssignVariableOp2@
stream_4/concat/ReadVariableOpstream_4/concat/ReadVariableOp2h
2stream_4/depthwise_conv2d_2/BiasAdd/ReadVariableOp2stream_4/depthwise_conv2d_2/BiasAdd/ReadVariableOp2l
4stream_4/depthwise_conv2d_2/depthwise/ReadVariableOp4stream_4/depthwise_conv2d_2/depthwise/ReadVariableOp26
stream_5/AssignVariableOpstream_5/AssignVariableOp2@
stream_5/concat/ReadVariableOpstream_5/concat/ReadVariableOp2h
2stream_5/depthwise_conv2d_3/BiasAdd/ReadVariableOp2stream_5/depthwise_conv2d_3/BiasAdd/ReadVariableOp2l
4stream_5/depthwise_conv2d_3/depthwise/ReadVariableOp4stream_5/depthwise_conv2d_3/depthwise/ReadVariableOp26
stream_6/AssignVariableOpstream_6/AssignVariableOp22
stream_6/ReadVariableOpstream_6/ReadVariableOp:O K
'
_output_shapes
:�
 
_user_specified_nameinputs
�

�
5__inference_batch_normalization_3_layer_call_fn_12673

inputs3
%streaming_batch_normalization_3_gamma:2
$streaming_batch_normalization_3_beta:9
+streaming_batch_normalization_3_moving_mean:=
/streaming_batch_normalization_3_moving_variance:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs%streaming_batch_normalization_3_gamma$streaming_batch_normalization_3_beta+streaming_batch_normalization_3_moving_mean/streaming_batch_normalization_3_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_10603�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�

�
3__inference_batch_normalization_layer_call_fn_12369

inputs1
#streaming_batch_normalization_gamma:0
"streaming_batch_normalization_beta:7
)streaming_batch_normalization_moving_mean:;
-streaming_batch_normalization_moving_variance:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs#streaming_batch_normalization_gamma"streaming_batch_normalization_beta)streaming_batch_normalization_moving_mean-streaming_batch_normalization_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_10283�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�v
�
@__inference_model_layer_call_and_return_conditional_losses_10968

iegm_input8
stream_streaming_stream_states:.
stream_stream_kernel: 
stream_stream_bias:E
7batch_normalization_streaming_batch_normalization_gamma:D
6batch_normalization_streaming_batch_normalization_beta:K
=batch_normalization_streaming_batch_normalization_moving_mean:O
Abatch_normalization_streaming_batch_normalization_moving_variance:<
"stream_1_streaming_stream_1_states:<
"stream_1_stream_1_depthwise_kernel:$
stream_1_stream_1_bias:I
;batch_normalization_1_streaming_batch_normalization_1_gamma:H
:batch_normalization_1_streaming_batch_normalization_1_beta:O
Abatch_normalization_1_streaming_batch_normalization_1_moving_mean:S
Ebatch_normalization_1_streaming_batch_normalization_1_moving_variance:<
"stream_2_streaming_stream_2_states:<
"stream_2_stream_2_depthwise_kernel:$
stream_2_stream_2_bias:I
;batch_normalization_2_streaming_batch_normalization_2_gamma:H
:batch_normalization_2_streaming_batch_normalization_2_beta:O
Abatch_normalization_2_streaming_batch_normalization_2_moving_mean:S
Ebatch_normalization_2_streaming_batch_normalization_2_moving_variance:<
"stream_3_streaming_stream_3_states:<
"stream_4_streaming_stream_4_states:<
"stream_4_stream_4_depthwise_kernel:$
stream_4_stream_4_bias:I
;batch_normalization_3_streaming_batch_normalization_3_gamma:H
:batch_normalization_3_streaming_batch_normalization_3_beta:O
Abatch_normalization_3_streaming_batch_normalization_3_moving_mean:S
Ebatch_normalization_3_streaming_batch_normalization_3_moving_variance:<
"stream_5_streaming_stream_5_states:<
"stream_5_stream_5_depthwise_kernel:$
stream_5_stream_5_bias:I
;batch_normalization_4_streaming_batch_normalization_4_gamma:H
:batch_normalization_4_streaming_batch_normalization_4_beta:O
Abatch_normalization_4_streaming_batch_normalization_4_moving_mean:S
Ebatch_normalization_4_streaming_batch_normalization_4_moving_variance:<
"stream_6_streaming_stream_6_states:.
dense_streaming_dense_kernel:0
(
dense_streaming_dense_bias:
2
 dense_1_streaming_dense_1_kernel:
,
dense_1_streaming_dense_1_bias:
identity��+batch_normalization/StatefulPartitionedCall�-batch_normalization_1/StatefulPartitionedCall�-batch_normalization_2/StatefulPartitionedCall�-batch_normalization_3/StatefulPartitionedCall�-batch_normalization_4/StatefulPartitionedCall�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dropout/StatefulPartitionedCall�stream/StatefulPartitionedCall� stream_1/StatefulPartitionedCall� stream_2/StatefulPartitionedCall� stream_3/StatefulPartitionedCall� stream_4/StatefulPartitionedCall� stream_5/StatefulPartitionedCall� stream_6/StatefulPartitionedCall�
max_pooling2d/PartitionedCallPartitionedCall
iegm_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:H* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_10222�
stream/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0stream_streaming_stream_statesstream_stream_kernelstream_stream_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_stream_layer_call_and_return_conditional_losses_10725�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'stream/StatefulPartitionedCall:output:07batch_normalization_streaming_batch_normalization_gamma6batch_normalization_streaming_batch_normalization_beta=batch_normalization_streaming_batch_normalization_moving_meanAbatch_normalization_streaming_batch_normalization_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_10283�
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:$* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_10740�
 stream_1/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0"stream_1_streaming_stream_1_states"stream_1_stream_1_depthwise_kernelstream_1_stream_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_stream_1_layer_call_and_return_conditional_losses_10762�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)stream_1/StatefulPartitionedCall:output:0;batch_normalization_1_streaming_batch_normalization_1_gamma:batch_normalization_1_streaming_batch_normalization_1_betaAbatch_normalization_1_streaming_batch_normalization_1_moving_meanEbatch_normalization_1_streaming_batch_normalization_1_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_10375�
activation_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_10777�
 stream_2/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0"stream_2_streaming_stream_2_states"stream_2_stream_2_depthwise_kernelstream_2_stream_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_stream_2_layer_call_and_return_conditional_losses_10799�
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)stream_2/StatefulPartitionedCall:output:0;batch_normalization_2_streaming_batch_normalization_2_gamma:batch_normalization_2_streaming_batch_normalization_2_betaAbatch_normalization_2_streaming_batch_normalization_2_moving_meanEbatch_normalization_2_streaming_batch_normalization_2_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_10467�
activation_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_activation_2_layer_call_and_return_conditional_losses_10814�
 stream_3/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:0"stream_3_streaming_stream_3_states*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_stream_3_layer_call_and_return_conditional_losses_10829�
 stream_4/StatefulPartitionedCallStatefulPartitionedCall)stream_3/StatefulPartitionedCall:output:0"stream_4_streaming_stream_4_states"stream_4_stream_4_depthwise_kernelstream_4_stream_4_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_stream_4_layer_call_and_return_conditional_losses_10852�
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)stream_4/StatefulPartitionedCall:output:0;batch_normalization_3_streaming_batch_normalization_3_gamma:batch_normalization_3_streaming_batch_normalization_3_betaAbatch_normalization_3_streaming_batch_normalization_3_moving_meanEbatch_normalization_3_streaming_batch_normalization_3_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_10576�
activation_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_activation_3_layer_call_and_return_conditional_losses_10867�
 stream_5/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0"stream_5_streaming_stream_5_states"stream_5_stream_5_depthwise_kernelstream_5_stream_5_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_stream_5_layer_call_and_return_conditional_losses_10889�
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall)stream_5/StatefulPartitionedCall:output:0;batch_normalization_4_streaming_batch_normalization_4_gamma:batch_normalization_4_streaming_batch_normalization_4_betaAbatch_normalization_4_streaming_batch_normalization_4_moving_meanEbatch_normalization_4_streaming_batch_normalization_4_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_10668�
activation_4/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_activation_4_layer_call_and_return_conditional_losses_10904�
dropout/StatefulPartitionedCallStatefulPartitionedCall%activation_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_10918�
 stream_6/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0"stream_6_streaming_stream_6_states*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_stream_6_layer_call_and_return_conditional_losses_10934�
dense/StatefulPartitionedCallStatefulPartitionedCall)stream_6/StatefulPartitionedCall:output:0dense_streaming_dense_kerneldense_streaming_dense_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_10948�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0 dense_1_streaming_dense_1_kerneldense_1_streaming_dense_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_10963n
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:�
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall^stream/StatefulPartitionedCall!^stream_1/StatefulPartitionedCall!^stream_2/StatefulPartitionedCall!^stream_3/StatefulPartitionedCall!^stream_4/StatefulPartitionedCall!^stream_5/StatefulPartitionedCall!^stream_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*x
_input_shapesg
e:�: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2@
stream/StatefulPartitionedCallstream/StatefulPartitionedCall2D
 stream_1/StatefulPartitionedCall stream_1/StatefulPartitionedCall2D
 stream_2/StatefulPartitionedCall stream_2/StatefulPartitionedCall2D
 stream_3/StatefulPartitionedCall stream_3/StatefulPartitionedCall2D
 stream_4/StatefulPartitionedCall stream_4/StatefulPartitionedCall2D
 stream_5/StatefulPartitionedCall stream_5/StatefulPartitionedCall2D
 stream_6/StatefulPartitionedCall stream_6/StatefulPartitionedCall:S O
'
_output_shapes
:�
$
_user_specified_name
IEGM_input
�
�
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_10467

inputsB
4readvariableop_streaming_batch_normalization_2_gamma:C
5readvariableop_1_streaming_batch_normalization_2_beta:Y
Kfusedbatchnormv3_readvariableop_streaming_batch_normalization_2_moving_mean:_
Qfusedbatchnormv3_readvariableop_1_streaming_batch_normalization_2_moving_variance:
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1
ReadVariableOpReadVariableOp4readvariableop_streaming_batch_normalization_2_gamma*
_output_shapes
:*
dtype0�
ReadVariableOp_1ReadVariableOp5readvariableop_1_streaming_batch_normalization_2_beta*
_output_shapes
:*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOpKfusedbatchnormv3_readvariableop_streaming_batch_normalization_2_moving_mean*
_output_shapes
:*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpQfusedbatchnormv3_readvariableop_1_streaming_batch_normalization_2_moving_variance*
_output_shapes
:*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%��'7*
exponential_avg_factor%fff?�
AssignNewValueAssignVariableOpKfusedbatchnormv3_readvariableop_streaming_batch_normalization_2_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOpQfusedbatchnormv3_readvariableop_1_streaming_batch_normalization_2_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
˵
�
!__inference__traced_restore_13306
file_prefixB
(assignvariableop_streaming_stream_states:D
6assignvariableop_1_streaming_batch_normalization_gamma:C
5assignvariableop_2_streaming_batch_normalization_beta:J
<assignvariableop_3_streaming_batch_normalization_moving_mean:N
@assignvariableop_4_streaming_batch_normalization_moving_variance:F
,assignvariableop_5_streaming_stream_1_states:F
8assignvariableop_6_streaming_batch_normalization_1_gamma:E
7assignvariableop_7_streaming_batch_normalization_1_beta:L
>assignvariableop_8_streaming_batch_normalization_1_moving_mean:P
Bassignvariableop_9_streaming_batch_normalization_1_moving_variance:G
-assignvariableop_10_streaming_stream_2_states:G
9assignvariableop_11_streaming_batch_normalization_2_gamma:F
8assignvariableop_12_streaming_batch_normalization_2_beta:M
?assignvariableop_13_streaming_batch_normalization_2_moving_mean:Q
Cassignvariableop_14_streaming_batch_normalization_2_moving_variance:G
-assignvariableop_15_streaming_stream_3_states:G
-assignvariableop_16_streaming_stream_4_states:G
9assignvariableop_17_streaming_batch_normalization_3_gamma:F
8assignvariableop_18_streaming_batch_normalization_3_beta:M
?assignvariableop_19_streaming_batch_normalization_3_moving_mean:Q
Cassignvariableop_20_streaming_batch_normalization_3_moving_variance:G
-assignvariableop_21_streaming_stream_5_states:G
9assignvariableop_22_streaming_batch_normalization_4_gamma:F
8assignvariableop_23_streaming_batch_normalization_4_beta:M
?assignvariableop_24_streaming_batch_normalization_4_moving_mean:Q
Cassignvariableop_25_streaming_batch_normalization_4_moving_variance:G
-assignvariableop_26_streaming_stream_6_states:<
*assignvariableop_27_streaming_dense_kernel:0
6
(assignvariableop_28_streaming_dense_bias:
>
,assignvariableop_29_streaming_dense_1_kernel:
8
*assignvariableop_30_streaming_dense_1_bias:;
!assignvariableop_31_stream_kernel:-
assignvariableop_32_stream_bias:G
-assignvariableop_33_stream_1_depthwise_kernel:/
!assignvariableop_34_stream_1_bias:G
-assignvariableop_35_stream_2_depthwise_kernel:/
!assignvariableop_36_stream_2_bias:G
-assignvariableop_37_stream_4_depthwise_kernel:/
!assignvariableop_38_stream_4_bias:G
-assignvariableop_39_stream_5_depthwise_kernel:/
!assignvariableop_40_stream_5_bias:
identity_42��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:**
dtype0*�
value�B�*B6layer_with_weights-0/states/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/states/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/states/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/states/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/states/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/states/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/states/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:**
dtype0*g
value^B\*B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::*8
dtypes.
,2*[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp(assignvariableop_streaming_stream_statesIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp6assignvariableop_1_streaming_batch_normalization_gammaIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp5assignvariableop_2_streaming_batch_normalization_betaIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp<assignvariableop_3_streaming_batch_normalization_moving_meanIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp@assignvariableop_4_streaming_batch_normalization_moving_varianceIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp,assignvariableop_5_streaming_stream_1_statesIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp8assignvariableop_6_streaming_batch_normalization_1_gammaIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp7assignvariableop_7_streaming_batch_normalization_1_betaIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp>assignvariableop_8_streaming_batch_normalization_1_moving_meanIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOpBassignvariableop_9_streaming_batch_normalization_1_moving_varianceIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp-assignvariableop_10_streaming_stream_2_statesIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp9assignvariableop_11_streaming_batch_normalization_2_gammaIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp8assignvariableop_12_streaming_batch_normalization_2_betaIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp?assignvariableop_13_streaming_batch_normalization_2_moving_meanIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOpCassignvariableop_14_streaming_batch_normalization_2_moving_varianceIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp-assignvariableop_15_streaming_stream_3_statesIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp-assignvariableop_16_streaming_stream_4_statesIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp9assignvariableop_17_streaming_batch_normalization_3_gammaIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp8assignvariableop_18_streaming_batch_normalization_3_betaIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp?assignvariableop_19_streaming_batch_normalization_3_moving_meanIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOpCassignvariableop_20_streaming_batch_normalization_3_moving_varianceIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp-assignvariableop_21_streaming_stream_5_statesIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp9assignvariableop_22_streaming_batch_normalization_4_gammaIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp8assignvariableop_23_streaming_batch_normalization_4_betaIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp?assignvariableop_24_streaming_batch_normalization_4_moving_meanIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOpCassignvariableop_25_streaming_batch_normalization_4_moving_varianceIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp-assignvariableop_26_streaming_stream_6_statesIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp*assignvariableop_27_streaming_dense_kernelIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp(assignvariableop_28_streaming_dense_biasIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp,assignvariableop_29_streaming_dense_1_kernelIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp*assignvariableop_30_streaming_dense_1_biasIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp!assignvariableop_31_stream_kernelIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOpassignvariableop_32_stream_biasIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp-assignvariableop_33_stream_1_depthwise_kernelIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp!assignvariableop_34_stream_1_biasIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp-assignvariableop_35_stream_2_depthwise_kernelIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOp!assignvariableop_36_stream_2_biasIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOp-assignvariableop_37_stream_4_depthwise_kernelIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOp!assignvariableop_38_stream_4_biasIdentity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOp-assignvariableop_39_stream_5_depthwise_kernelIdentity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOp!assignvariableop_40_stream_5_biasIdentity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_41Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_42IdentityIdentity_41:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_42Identity_42:output:0*g
_input_shapesV
T: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_40AssignVariableOp_402(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�

�
5__inference_batch_normalization_4_layer_call_fn_12756

inputs3
%streaming_batch_normalization_4_gamma:2
$streaming_batch_normalization_4_beta:9
+streaming_batch_normalization_4_moving_mean:=
/streaming_batch_normalization_4_moving_variance:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs%streaming_batch_normalization_4_gamma$streaming_batch_normalization_4_beta+streaming_batch_normalization_4_moving_mean/streaming_batch_normalization_4_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_10668�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_10695

inputsB
4readvariableop_streaming_batch_normalization_4_gamma:C
5readvariableop_1_streaming_batch_normalization_4_beta:Y
Kfusedbatchnormv3_readvariableop_streaming_batch_normalization_4_moving_mean:_
Qfusedbatchnormv3_readvariableop_1_streaming_batch_normalization_4_moving_variance:
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1
ReadVariableOpReadVariableOp4readvariableop_streaming_batch_normalization_4_gamma*
_output_shapes
:*
dtype0�
ReadVariableOp_1ReadVariableOp5readvariableop_1_streaming_batch_normalization_4_beta*
_output_shapes
:*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOpKfusedbatchnormv3_readvariableop_streaming_batch_normalization_4_moving_mean*
_output_shapes
:*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpQfusedbatchnormv3_readvariableop_1_streaming_batch_normalization_4_moving_variance*
_output_shapes
:*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%��'7*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
C__inference_stream_6_layer_call_and_return_conditional_losses_10934

inputsB
(readvariableop_streaming_stream_6_states:
identity��AssignVariableOp�ReadVariableOp
ReadVariableOpReadVariableOp(readvariableop_streaming_stream_6_states*&
_output_shapes
:*
dtype0h
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           j
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           j
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         �
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*&
_output_shapes
:*

begin_mask*
end_maskM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatConcatV2strided_slice:output:0inputsconcat/axis:output:0*
N*
T0*&
_output_shapes
:�
AssignVariableOpAssignVariableOp(readvariableop_streaming_stream_6_statesconcat:output:0^ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(q
flatten/ConstConst^AssignVariableOp*
_output_shapes
:*
dtype0*
valueB"����0   l
flatten/ReshapeReshapeconcat:output:0flatten/Const:output:0*
T0*
_output_shapes

:0^
IdentityIdentityflatten/Reshape:output:0^NoOp*
T0*
_output_shapes

:0j
NoOpNoOp^AssignVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:: 2$
AssignVariableOpAssignVariableOp2 
ReadVariableOpReadVariableOp:N J
&
_output_shapes
:
 
_user_specified_nameinputs
�	
�
B__inference_dense_1_layer_call_and_return_conditional_losses_12894

inputs@
.matmul_readvariableop_streaming_dense_1_kernel:
;
-biasadd_readvariableop_streaming_dense_1_bias:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOp.matmul_readvariableop_streaming_dense_1_kernel*
_output_shapes

:
*
dtype0`
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:�
BiasAdd/ReadVariableOpReadVariableOp-biasadd_readvariableop_streaming_dense_1_bias*
_output_shapes
:*
dtype0m
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:M
SoftmaxSoftmaxBiasAdd:output:0*
T0*
_output_shapes

:W
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*
_output_shapes

:w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*!
_input_shapes
:
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:F B

_output_shapes

:

 
_user_specified_nameinputs
�
�
(__inference_stream_2_layer_call_fn_12524

inputs3
streaming_stream_2_states:3
stream_2_depthwise_kernel:
stream_2_bias:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstreaming_stream_2_statesstream_2_depthwise_kernelstream_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_stream_2_layer_call_and_return_conditional_losses_10799n
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*&
_output_shapes
:`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*+
_input_shapes
:: : : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
:
 
_user_specified_nameinputs
�
�
C__inference_stream_4_layer_call_and_return_conditional_losses_12655

inputsI
/concat_readvariableop_streaming_stream_4_states:_
Edepthwise_conv2d_2_depthwise_readvariableop_stream_4_depthwise_kernel:E
7depthwise_conv2d_2_biasadd_readvariableop_stream_4_bias:
identity��AssignVariableOp�concat/ReadVariableOp�)depthwise_conv2d_2/BiasAdd/ReadVariableOp�+depthwise_conv2d_2/depthwise/ReadVariableOp�
concat/ReadVariableOpReadVariableOp/concat_readvariableop_streaming_stream_4_states*&
_output_shapes
:*
dtype0M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatConcatV2concat/ReadVariableOp:value:0inputsconcat/axis:output:0*
N*
T0*&
_output_shapes
:	h
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ����    j
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            j
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         �
strided_sliceStridedSliceconcat:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*&
_output_shapes
:*

begin_mask*
end_mask�
AssignVariableOpAssignVariableOp/concat_readvariableop_streaming_stream_4_statesstrided_slice:output:0^concat/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
+depthwise_conv2d_2/depthwise/ReadVariableOpReadVariableOpEdepthwise_conv2d_2_depthwise_readvariableop_stream_4_depthwise_kernel^AssignVariableOp*&
_output_shapes
:*
dtype0�
"depthwise_conv2d_2/depthwise/ShapeConst^AssignVariableOp*
_output_shapes
:*
dtype0*%
valueB"            �
*depthwise_conv2d_2/depthwise/dilation_rateConst^AssignVariableOp*
_output_shapes
:*
dtype0*
valueB"      �
depthwise_conv2d_2/depthwiseDepthwiseConv2dNativeconcat:output:03depthwise_conv2d_2/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
:*
paddingVALID*
strides
�
)depthwise_conv2d_2/BiasAdd/ReadVariableOpReadVariableOp7depthwise_conv2d_2_biasadd_readvariableop_stream_4_bias^AssignVariableOp*
_output_shapes
:*
dtype0�
depthwise_conv2d_2/BiasAddBiasAdd%depthwise_conv2d_2/depthwise:output:01depthwise_conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:q
IdentityIdentity#depthwise_conv2d_2/BiasAdd:output:0^NoOp*
T0*&
_output_shapes
:�
NoOpNoOp^AssignVariableOp^concat/ReadVariableOp*^depthwise_conv2d_2/BiasAdd/ReadVariableOp,^depthwise_conv2d_2/depthwise/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*+
_input_shapes
:: : : 2$
AssignVariableOpAssignVariableOp2.
concat/ReadVariableOpconcat/ReadVariableOp2V
)depthwise_conv2d_2/BiasAdd/ReadVariableOp)depthwise_conv2d_2/BiasAdd/ReadVariableOp2Z
+depthwise_conv2d_2/depthwise/ReadVariableOp+depthwise_conv2d_2/depthwise/ReadVariableOp:N J
&
_output_shapes
:
 
_user_specified_nameinputs
��
�1
 __inference__wrapped_model_10208

iegm_inputT
:model_stream_concat_readvariableop_streaming_stream_states:Q
7model_stream_conv2d_conv2d_readvariableop_stream_kernel:D
6model_stream_conv2d_biasadd_readvariableop_stream_bias:Z
Lmodel_batch_normalization_readvariableop_streaming_batch_normalization_gamma:[
Mmodel_batch_normalization_readvariableop_1_streaming_batch_normalization_beta:q
cmodel_batch_normalization_fusedbatchnormv3_readvariableop_streaming_batch_normalization_moving_mean:w
imodel_batch_normalization_fusedbatchnormv3_readvariableop_1_streaming_batch_normalization_moving_variance:X
>model_stream_1_concat_readvariableop_streaming_stream_1_states:l
Rmodel_stream_1_depthwise_conv2d_depthwise_readvariableop_stream_1_depthwise_kernel:R
Dmodel_stream_1_depthwise_conv2d_biasadd_readvariableop_stream_1_bias:^
Pmodel_batch_normalization_1_readvariableop_streaming_batch_normalization_1_gamma:_
Qmodel_batch_normalization_1_readvariableop_1_streaming_batch_normalization_1_beta:u
gmodel_batch_normalization_1_fusedbatchnormv3_readvariableop_streaming_batch_normalization_1_moving_mean:{
mmodel_batch_normalization_1_fusedbatchnormv3_readvariableop_1_streaming_batch_normalization_1_moving_variance:X
>model_stream_2_concat_readvariableop_streaming_stream_2_states:n
Tmodel_stream_2_depthwise_conv2d_1_depthwise_readvariableop_stream_2_depthwise_kernel:T
Fmodel_stream_2_depthwise_conv2d_1_biasadd_readvariableop_stream_2_bias:^
Pmodel_batch_normalization_2_readvariableop_streaming_batch_normalization_2_gamma:_
Qmodel_batch_normalization_2_readvariableop_1_streaming_batch_normalization_2_beta:u
gmodel_batch_normalization_2_fusedbatchnormv3_readvariableop_streaming_batch_normalization_2_moving_mean:{
mmodel_batch_normalization_2_fusedbatchnormv3_readvariableop_1_streaming_batch_normalization_2_moving_variance:X
>model_stream_3_concat_readvariableop_streaming_stream_3_states:X
>model_stream_4_concat_readvariableop_streaming_stream_4_states:n
Tmodel_stream_4_depthwise_conv2d_2_depthwise_readvariableop_stream_4_depthwise_kernel:T
Fmodel_stream_4_depthwise_conv2d_2_biasadd_readvariableop_stream_4_bias:^
Pmodel_batch_normalization_3_readvariableop_streaming_batch_normalization_3_gamma:_
Qmodel_batch_normalization_3_readvariableop_1_streaming_batch_normalization_3_beta:u
gmodel_batch_normalization_3_fusedbatchnormv3_readvariableop_streaming_batch_normalization_3_moving_mean:{
mmodel_batch_normalization_3_fusedbatchnormv3_readvariableop_1_streaming_batch_normalization_3_moving_variance:X
>model_stream_5_concat_readvariableop_streaming_stream_5_states:n
Tmodel_stream_5_depthwise_conv2d_3_depthwise_readvariableop_stream_5_depthwise_kernel:T
Fmodel_stream_5_depthwise_conv2d_3_biasadd_readvariableop_stream_5_bias:^
Pmodel_batch_normalization_4_readvariableop_streaming_batch_normalization_4_gamma:_
Qmodel_batch_normalization_4_readvariableop_1_streaming_batch_normalization_4_beta:u
gmodel_batch_normalization_4_fusedbatchnormv3_readvariableop_streaming_batch_normalization_4_moving_mean:{
mmodel_batch_normalization_4_fusedbatchnormv3_readvariableop_1_streaming_batch_normalization_4_moving_variance:Q
7model_stream_6_readvariableop_streaming_stream_6_states:J
8model_dense_matmul_readvariableop_streaming_dense_kernel:0
E
7model_dense_biasadd_readvariableop_streaming_dense_bias:
N
<model_dense_1_matmul_readvariableop_streaming_dense_1_kernel:
I
;model_dense_1_biasadd_readvariableop_streaming_dense_1_bias:
identity��9model/batch_normalization/FusedBatchNormV3/ReadVariableOp�;model/batch_normalization/FusedBatchNormV3/ReadVariableOp_1�(model/batch_normalization/ReadVariableOp�*model/batch_normalization/ReadVariableOp_1�;model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp�=model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1�*model/batch_normalization_1/ReadVariableOp�,model/batch_normalization_1/ReadVariableOp_1�;model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp�=model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1�*model/batch_normalization_2/ReadVariableOp�,model/batch_normalization_2/ReadVariableOp_1�;model/batch_normalization_3/FusedBatchNormV3/ReadVariableOp�=model/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1�*model/batch_normalization_3/ReadVariableOp�,model/batch_normalization_3/ReadVariableOp_1�;model/batch_normalization_4/FusedBatchNormV3/ReadVariableOp�=model/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1�*model/batch_normalization_4/ReadVariableOp�,model/batch_normalization_4/ReadVariableOp_1�"model/dense/BiasAdd/ReadVariableOp�!model/dense/MatMul/ReadVariableOp�$model/dense_1/BiasAdd/ReadVariableOp�#model/dense_1/MatMul/ReadVariableOp�model/stream/AssignVariableOp�"model/stream/concat/ReadVariableOp�*model/stream/conv2d/BiasAdd/ReadVariableOp�)model/stream/conv2d/Conv2D/ReadVariableOp�model/stream_1/AssignVariableOp�$model/stream_1/concat/ReadVariableOp�6model/stream_1/depthwise_conv2d/BiasAdd/ReadVariableOp�8model/stream_1/depthwise_conv2d/depthwise/ReadVariableOp�model/stream_2/AssignVariableOp�$model/stream_2/concat/ReadVariableOp�8model/stream_2/depthwise_conv2d_1/BiasAdd/ReadVariableOp�:model/stream_2/depthwise_conv2d_1/depthwise/ReadVariableOp�model/stream_3/AssignVariableOp�$model/stream_3/concat/ReadVariableOp�model/stream_4/AssignVariableOp�$model/stream_4/concat/ReadVariableOp�8model/stream_4/depthwise_conv2d_2/BiasAdd/ReadVariableOp�:model/stream_4/depthwise_conv2d_2/depthwise/ReadVariableOp�model/stream_5/AssignVariableOp�$model/stream_5/concat/ReadVariableOp�8model/stream_5/depthwise_conv2d_3/BiasAdd/ReadVariableOp�:model/stream_5/depthwise_conv2d_3/depthwise/ReadVariableOp�model/stream_6/AssignVariableOp�model/stream_6/ReadVariableOp�
model/max_pooling2d/MaxPoolMaxPool
iegm_input*&
_output_shapes
:H*
ksize
*
paddingVALID*
strides
�
"model/stream/concat/ReadVariableOpReadVariableOp:model_stream_concat_readvariableop_streaming_stream_states*&
_output_shapes
:*
dtype0Z
model/stream/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
model/stream/concatConcatV2*model/stream/concat/ReadVariableOp:value:0$model/max_pooling2d/MaxPool:output:0!model/stream/concat/axis:output:0*
N*
T0*&
_output_shapes
:Lu
 model/stream/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ����    w
"model/stream/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            w
"model/stream/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         �
model/stream/strided_sliceStridedSlicemodel/stream/concat:output:0)model/stream/strided_slice/stack:output:0+model/stream/strided_slice/stack_1:output:0+model/stream/strided_slice/stack_2:output:0*
Index0*
T0*&
_output_shapes
:*

begin_mask*
end_mask�
model/stream/AssignVariableOpAssignVariableOp:model_stream_concat_readvariableop_streaming_stream_states#model/stream/strided_slice:output:0#^model/stream/concat/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
)model/stream/conv2d/Conv2D/ReadVariableOpReadVariableOp7model_stream_conv2d_conv2d_readvariableop_stream_kernel^model/stream/AssignVariableOp*&
_output_shapes
:*
dtype0�
model/stream/conv2d/Conv2DConv2Dmodel/stream/concat:output:01model/stream/conv2d/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:$*
paddingVALID*
strides
�
*model/stream/conv2d/BiasAdd/ReadVariableOpReadVariableOp6model_stream_conv2d_biasadd_readvariableop_stream_bias^model/stream/AssignVariableOp*
_output_shapes
:*
dtype0�
model/stream/conv2d/BiasAddBiasAdd#model/stream/conv2d/Conv2D:output:02model/stream/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:$�
(model/batch_normalization/ReadVariableOpReadVariableOpLmodel_batch_normalization_readvariableop_streaming_batch_normalization_gamma*
_output_shapes
:*
dtype0�
*model/batch_normalization/ReadVariableOp_1ReadVariableOpMmodel_batch_normalization_readvariableop_1_streaming_batch_normalization_beta*
_output_shapes
:*
dtype0�
9model/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpcmodel_batch_normalization_fusedbatchnormv3_readvariableop_streaming_batch_normalization_moving_mean*
_output_shapes
:*
dtype0�
;model/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpimodel_batch_normalization_fusedbatchnormv3_readvariableop_1_streaming_batch_normalization_moving_variance*
_output_shapes
:*
dtype0�
*model/batch_normalization/FusedBatchNormV3FusedBatchNormV3$model/stream/conv2d/BiasAdd:output:00model/batch_normalization/ReadVariableOp:value:02model/batch_normalization/ReadVariableOp_1:value:0Amodel/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Cmodel/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:$:::::*
epsilon%��'7*
is_training( ~
model/activation/ReluRelu.model/batch_normalization/FusedBatchNormV3:y:0*
T0*&
_output_shapes
:$�
$model/stream_1/concat/ReadVariableOpReadVariableOp>model_stream_1_concat_readvariableop_streaming_stream_1_states*&
_output_shapes
:*
dtype0\
model/stream_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
model/stream_1/concatConcatV2,model/stream_1/concat/ReadVariableOp:value:0#model/activation/Relu:activations:0#model/stream_1/concat/axis:output:0*
N*
T0*&
_output_shapes
:'w
"model/stream_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ����    y
$model/stream_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            y
$model/stream_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         �
model/stream_1/strided_sliceStridedSlicemodel/stream_1/concat:output:0+model/stream_1/strided_slice/stack:output:0-model/stream_1/strided_slice/stack_1:output:0-model/stream_1/strided_slice/stack_2:output:0*
Index0*
T0*&
_output_shapes
:*

begin_mask*
end_mask�
model/stream_1/AssignVariableOpAssignVariableOp>model_stream_1_concat_readvariableop_streaming_stream_1_states%model/stream_1/strided_slice:output:0%^model/stream_1/concat/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
8model/stream_1/depthwise_conv2d/depthwise/ReadVariableOpReadVariableOpRmodel_stream_1_depthwise_conv2d_depthwise_readvariableop_stream_1_depthwise_kernel ^model/stream_1/AssignVariableOp*&
_output_shapes
:*
dtype0�
/model/stream_1/depthwise_conv2d/depthwise/ShapeConst ^model/stream_1/AssignVariableOp*
_output_shapes
:*
dtype0*%
valueB"            �
7model/stream_1/depthwise_conv2d/depthwise/dilation_rateConst ^model/stream_1/AssignVariableOp*
_output_shapes
:*
dtype0*
valueB"      �
)model/stream_1/depthwise_conv2d/depthwiseDepthwiseConv2dNativemodel/stream_1/concat:output:0@model/stream_1/depthwise_conv2d/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
:*
paddingVALID*
strides
�
6model/stream_1/depthwise_conv2d/BiasAdd/ReadVariableOpReadVariableOpDmodel_stream_1_depthwise_conv2d_biasadd_readvariableop_stream_1_bias ^model/stream_1/AssignVariableOp*
_output_shapes
:*
dtype0�
'model/stream_1/depthwise_conv2d/BiasAddBiasAdd2model/stream_1/depthwise_conv2d/depthwise:output:0>model/stream_1/depthwise_conv2d/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:�
*model/batch_normalization_1/ReadVariableOpReadVariableOpPmodel_batch_normalization_1_readvariableop_streaming_batch_normalization_1_gamma*
_output_shapes
:*
dtype0�
,model/batch_normalization_1/ReadVariableOp_1ReadVariableOpQmodel_batch_normalization_1_readvariableop_1_streaming_batch_normalization_1_beta*
_output_shapes
:*
dtype0�
;model/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpgmodel_batch_normalization_1_fusedbatchnormv3_readvariableop_streaming_batch_normalization_1_moving_mean*
_output_shapes
:*
dtype0�
=model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpmmodel_batch_normalization_1_fusedbatchnormv3_readvariableop_1_streaming_batch_normalization_1_moving_variance*
_output_shapes
:*
dtype0�
,model/batch_normalization_1/FusedBatchNormV3FusedBatchNormV30model/stream_1/depthwise_conv2d/BiasAdd:output:02model/batch_normalization_1/ReadVariableOp:value:04model/batch_normalization_1/ReadVariableOp_1:value:0Cmodel/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Emodel/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.::::::*
epsilon%��'7*
is_training( �
model/activation_1/ReluRelu0model/batch_normalization_1/FusedBatchNormV3:y:0*
T0*&
_output_shapes
:�
$model/stream_2/concat/ReadVariableOpReadVariableOp>model_stream_2_concat_readvariableop_streaming_stream_2_states*&
_output_shapes
:*
dtype0\
model/stream_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
model/stream_2/concatConcatV2,model/stream_2/concat/ReadVariableOp:value:0%model/activation_1/Relu:activations:0#model/stream_2/concat/axis:output:0*
N*
T0*&
_output_shapes
:w
"model/stream_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ����    y
$model/stream_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            y
$model/stream_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         �
model/stream_2/strided_sliceStridedSlicemodel/stream_2/concat:output:0+model/stream_2/strided_slice/stack:output:0-model/stream_2/strided_slice/stack_1:output:0-model/stream_2/strided_slice/stack_2:output:0*
Index0*
T0*&
_output_shapes
:*

begin_mask*
end_mask�
model/stream_2/AssignVariableOpAssignVariableOp>model_stream_2_concat_readvariableop_streaming_stream_2_states%model/stream_2/strided_slice:output:0%^model/stream_2/concat/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
:model/stream_2/depthwise_conv2d_1/depthwise/ReadVariableOpReadVariableOpTmodel_stream_2_depthwise_conv2d_1_depthwise_readvariableop_stream_2_depthwise_kernel ^model/stream_2/AssignVariableOp*&
_output_shapes
:*
dtype0�
1model/stream_2/depthwise_conv2d_1/depthwise/ShapeConst ^model/stream_2/AssignVariableOp*
_output_shapes
:*
dtype0*%
valueB"            �
9model/stream_2/depthwise_conv2d_1/depthwise/dilation_rateConst ^model/stream_2/AssignVariableOp*
_output_shapes
:*
dtype0*
valueB"      �
+model/stream_2/depthwise_conv2d_1/depthwiseDepthwiseConv2dNativemodel/stream_2/concat:output:0Bmodel/stream_2/depthwise_conv2d_1/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
:*
paddingVALID*
strides
�
8model/stream_2/depthwise_conv2d_1/BiasAdd/ReadVariableOpReadVariableOpFmodel_stream_2_depthwise_conv2d_1_biasadd_readvariableop_stream_2_bias ^model/stream_2/AssignVariableOp*
_output_shapes
:*
dtype0�
)model/stream_2/depthwise_conv2d_1/BiasAddBiasAdd4model/stream_2/depthwise_conv2d_1/depthwise:output:0@model/stream_2/depthwise_conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:�
*model/batch_normalization_2/ReadVariableOpReadVariableOpPmodel_batch_normalization_2_readvariableop_streaming_batch_normalization_2_gamma*
_output_shapes
:*
dtype0�
,model/batch_normalization_2/ReadVariableOp_1ReadVariableOpQmodel_batch_normalization_2_readvariableop_1_streaming_batch_normalization_2_beta*
_output_shapes
:*
dtype0�
;model/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpgmodel_batch_normalization_2_fusedbatchnormv3_readvariableop_streaming_batch_normalization_2_moving_mean*
_output_shapes
:*
dtype0�
=model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpmmodel_batch_normalization_2_fusedbatchnormv3_readvariableop_1_streaming_batch_normalization_2_moving_variance*
_output_shapes
:*
dtype0�
,model/batch_normalization_2/FusedBatchNormV3FusedBatchNormV32model/stream_2/depthwise_conv2d_1/BiasAdd:output:02model/batch_normalization_2/ReadVariableOp:value:04model/batch_normalization_2/ReadVariableOp_1:value:0Cmodel/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Emodel/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.::::::*
epsilon%��'7*
is_training( �
model/activation_2/ReluRelu0model/batch_normalization_2/FusedBatchNormV3:y:0*
T0*&
_output_shapes
:�
$model/stream_3/concat/ReadVariableOpReadVariableOp>model_stream_3_concat_readvariableop_streaming_stream_3_states*&
_output_shapes
:*
dtype0\
model/stream_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
model/stream_3/concatConcatV2,model/stream_3/concat/ReadVariableOp:value:0%model/activation_2/Relu:activations:0#model/stream_3/concat/axis:output:0*
N*
T0*&
_output_shapes
:w
"model/stream_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ����    y
$model/stream_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            y
$model/stream_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         �
model/stream_3/strided_sliceStridedSlicemodel/stream_3/concat:output:0+model/stream_3/strided_slice/stack:output:0-model/stream_3/strided_slice/stack_1:output:0-model/stream_3/strided_slice/stack_2:output:0*
Index0*
T0*&
_output_shapes
:*

begin_mask*
end_mask�
model/stream_3/AssignVariableOpAssignVariableOp>model_stream_3_concat_readvariableop_streaming_stream_3_states%model/stream_3/strided_slice:output:0%^model/stream_3/concat/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
(model/stream_3/average_pooling2d/AvgPoolAvgPoolmodel/stream_3/concat:output:0 ^model/stream_3/AssignVariableOp*
T0*&
_output_shapes
:*
ksize
*
paddingVALID*
strides
�
$model/stream_4/concat/ReadVariableOpReadVariableOp>model_stream_4_concat_readvariableop_streaming_stream_4_states*&
_output_shapes
:*
dtype0\
model/stream_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
model/stream_4/concatConcatV2,model/stream_4/concat/ReadVariableOp:value:01model/stream_3/average_pooling2d/AvgPool:output:0#model/stream_4/concat/axis:output:0*
N*
T0*&
_output_shapes
:	w
"model/stream_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ����    y
$model/stream_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            y
$model/stream_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         �
model/stream_4/strided_sliceStridedSlicemodel/stream_4/concat:output:0+model/stream_4/strided_slice/stack:output:0-model/stream_4/strided_slice/stack_1:output:0-model/stream_4/strided_slice/stack_2:output:0*
Index0*
T0*&
_output_shapes
:*

begin_mask*
end_mask�
model/stream_4/AssignVariableOpAssignVariableOp>model_stream_4_concat_readvariableop_streaming_stream_4_states%model/stream_4/strided_slice:output:0%^model/stream_4/concat/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
:model/stream_4/depthwise_conv2d_2/depthwise/ReadVariableOpReadVariableOpTmodel_stream_4_depthwise_conv2d_2_depthwise_readvariableop_stream_4_depthwise_kernel ^model/stream_4/AssignVariableOp*&
_output_shapes
:*
dtype0�
1model/stream_4/depthwise_conv2d_2/depthwise/ShapeConst ^model/stream_4/AssignVariableOp*
_output_shapes
:*
dtype0*%
valueB"            �
9model/stream_4/depthwise_conv2d_2/depthwise/dilation_rateConst ^model/stream_4/AssignVariableOp*
_output_shapes
:*
dtype0*
valueB"      �
+model/stream_4/depthwise_conv2d_2/depthwiseDepthwiseConv2dNativemodel/stream_4/concat:output:0Bmodel/stream_4/depthwise_conv2d_2/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
:*
paddingVALID*
strides
�
8model/stream_4/depthwise_conv2d_2/BiasAdd/ReadVariableOpReadVariableOpFmodel_stream_4_depthwise_conv2d_2_biasadd_readvariableop_stream_4_bias ^model/stream_4/AssignVariableOp*
_output_shapes
:*
dtype0�
)model/stream_4/depthwise_conv2d_2/BiasAddBiasAdd4model/stream_4/depthwise_conv2d_2/depthwise:output:0@model/stream_4/depthwise_conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:�
*model/batch_normalization_3/ReadVariableOpReadVariableOpPmodel_batch_normalization_3_readvariableop_streaming_batch_normalization_3_gamma*
_output_shapes
:*
dtype0�
,model/batch_normalization_3/ReadVariableOp_1ReadVariableOpQmodel_batch_normalization_3_readvariableop_1_streaming_batch_normalization_3_beta*
_output_shapes
:*
dtype0�
;model/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpgmodel_batch_normalization_3_fusedbatchnormv3_readvariableop_streaming_batch_normalization_3_moving_mean*
_output_shapes
:*
dtype0�
=model/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpmmodel_batch_normalization_3_fusedbatchnormv3_readvariableop_1_streaming_batch_normalization_3_moving_variance*
_output_shapes
:*
dtype0�
,model/batch_normalization_3/FusedBatchNormV3FusedBatchNormV32model/stream_4/depthwise_conv2d_2/BiasAdd:output:02model/batch_normalization_3/ReadVariableOp:value:04model/batch_normalization_3/ReadVariableOp_1:value:0Cmodel/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Emodel/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.::::::*
epsilon%��'7*
is_training( �
model/activation_3/ReluRelu0model/batch_normalization_3/FusedBatchNormV3:y:0*
T0*&
_output_shapes
:�
$model/stream_5/concat/ReadVariableOpReadVariableOp>model_stream_5_concat_readvariableop_streaming_stream_5_states*&
_output_shapes
:*
dtype0\
model/stream_5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
model/stream_5/concatConcatV2,model/stream_5/concat/ReadVariableOp:value:0%model/activation_3/Relu:activations:0#model/stream_5/concat/axis:output:0*
N*
T0*&
_output_shapes
:w
"model/stream_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ����    y
$model/stream_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            y
$model/stream_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         �
model/stream_5/strided_sliceStridedSlicemodel/stream_5/concat:output:0+model/stream_5/strided_slice/stack:output:0-model/stream_5/strided_slice/stack_1:output:0-model/stream_5/strided_slice/stack_2:output:0*
Index0*
T0*&
_output_shapes
:*

begin_mask*
end_mask�
model/stream_5/AssignVariableOpAssignVariableOp>model_stream_5_concat_readvariableop_streaming_stream_5_states%model/stream_5/strided_slice:output:0%^model/stream_5/concat/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
:model/stream_5/depthwise_conv2d_3/depthwise/ReadVariableOpReadVariableOpTmodel_stream_5_depthwise_conv2d_3_depthwise_readvariableop_stream_5_depthwise_kernel ^model/stream_5/AssignVariableOp*&
_output_shapes
:*
dtype0�
1model/stream_5/depthwise_conv2d_3/depthwise/ShapeConst ^model/stream_5/AssignVariableOp*
_output_shapes
:*
dtype0*%
valueB"            �
9model/stream_5/depthwise_conv2d_3/depthwise/dilation_rateConst ^model/stream_5/AssignVariableOp*
_output_shapes
:*
dtype0*
valueB"      �
+model/stream_5/depthwise_conv2d_3/depthwiseDepthwiseConv2dNativemodel/stream_5/concat:output:0Bmodel/stream_5/depthwise_conv2d_3/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
:*
paddingVALID*
strides
�
8model/stream_5/depthwise_conv2d_3/BiasAdd/ReadVariableOpReadVariableOpFmodel_stream_5_depthwise_conv2d_3_biasadd_readvariableop_stream_5_bias ^model/stream_5/AssignVariableOp*
_output_shapes
:*
dtype0�
)model/stream_5/depthwise_conv2d_3/BiasAddBiasAdd4model/stream_5/depthwise_conv2d_3/depthwise:output:0@model/stream_5/depthwise_conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:�
*model/batch_normalization_4/ReadVariableOpReadVariableOpPmodel_batch_normalization_4_readvariableop_streaming_batch_normalization_4_gamma*
_output_shapes
:*
dtype0�
,model/batch_normalization_4/ReadVariableOp_1ReadVariableOpQmodel_batch_normalization_4_readvariableop_1_streaming_batch_normalization_4_beta*
_output_shapes
:*
dtype0�
;model/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpgmodel_batch_normalization_4_fusedbatchnormv3_readvariableop_streaming_batch_normalization_4_moving_mean*
_output_shapes
:*
dtype0�
=model/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpmmodel_batch_normalization_4_fusedbatchnormv3_readvariableop_1_streaming_batch_normalization_4_moving_variance*
_output_shapes
:*
dtype0�
,model/batch_normalization_4/FusedBatchNormV3FusedBatchNormV32model/stream_5/depthwise_conv2d_3/BiasAdd:output:02model/batch_normalization_4/ReadVariableOp:value:04model/batch_normalization_4/ReadVariableOp_1:value:0Cmodel/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0Emodel/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.::::::*
epsilon%��'7*
is_training( �
model/activation_4/ReluRelu0model/batch_normalization_4/FusedBatchNormV3:y:0*
T0*&
_output_shapes
:z
model/dropout/IdentityIdentity%model/activation_4/Relu:activations:0*
T0*&
_output_shapes
:�
model/stream_6/ReadVariableOpReadVariableOp7model_stream_6_readvariableop_streaming_stream_6_states*&
_output_shapes
:*
dtype0w
"model/stream_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           y
$model/stream_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           y
$model/stream_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         �
model/stream_6/strided_sliceStridedSlice%model/stream_6/ReadVariableOp:value:0+model/stream_6/strided_slice/stack:output:0-model/stream_6/strided_slice/stack_1:output:0-model/stream_6/strided_slice/stack_2:output:0*
Index0*
T0*&
_output_shapes
:*

begin_mask*
end_mask\
model/stream_6/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
model/stream_6/concatConcatV2%model/stream_6/strided_slice:output:0model/dropout/Identity:output:0#model/stream_6/concat/axis:output:0*
N*
T0*&
_output_shapes
:�
model/stream_6/AssignVariableOpAssignVariableOp7model_stream_6_readvariableop_streaming_stream_6_statesmodel/stream_6/concat:output:0^model/stream_6/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
model/stream_6/flatten/ConstConst ^model/stream_6/AssignVariableOp*
_output_shapes
:*
dtype0*
valueB"����0   �
model/stream_6/flatten/ReshapeReshapemodel/stream_6/concat:output:0%model/stream_6/flatten/Const:output:0*
T0*
_output_shapes

:0�
!model/dense/MatMul/ReadVariableOpReadVariableOp8model_dense_matmul_readvariableop_streaming_dense_kernel*
_output_shapes

:0
*
dtype0�
model/dense/MatMulMatMul'model/stream_6/flatten/Reshape:output:0)model/dense/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:
�
"model/dense/BiasAdd/ReadVariableOpReadVariableOp7model_dense_biasadd_readvariableop_streaming_dense_bias*
_output_shapes
:
*
dtype0�
model/dense/BiasAddBiasAddmodel/dense/MatMul:product:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:
_
model/dense/ReluRelumodel/dense/BiasAdd:output:0*
T0*
_output_shapes

:
�
#model/dense_1/MatMul/ReadVariableOpReadVariableOp<model_dense_1_matmul_readvariableop_streaming_dense_1_kernel*
_output_shapes

:
*
dtype0�
model/dense_1/MatMulMatMulmodel/dense/Relu:activations:0+model/dense_1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:�
$model/dense_1/BiasAdd/ReadVariableOpReadVariableOp;model_dense_1_biasadd_readvariableop_streaming_dense_1_bias*
_output_shapes
:*
dtype0�
model/dense_1/BiasAddBiasAddmodel/dense_1/MatMul:product:0,model/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:i
model/dense_1/SoftmaxSoftmaxmodel/dense_1/BiasAdd:output:0*
T0*
_output_shapes

:e
IdentityIdentitymodel/dense_1/Softmax:softmax:0^NoOp*
T0*
_output_shapes

:�
NoOpNoOp:^model/batch_normalization/FusedBatchNormV3/ReadVariableOp<^model/batch_normalization/FusedBatchNormV3/ReadVariableOp_1)^model/batch_normalization/ReadVariableOp+^model/batch_normalization/ReadVariableOp_1<^model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp>^model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1+^model/batch_normalization_1/ReadVariableOp-^model/batch_normalization_1/ReadVariableOp_1<^model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp>^model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1+^model/batch_normalization_2/ReadVariableOp-^model/batch_normalization_2/ReadVariableOp_1<^model/batch_normalization_3/FusedBatchNormV3/ReadVariableOp>^model/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1+^model/batch_normalization_3/ReadVariableOp-^model/batch_normalization_3/ReadVariableOp_1<^model/batch_normalization_4/FusedBatchNormV3/ReadVariableOp>^model/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1+^model/batch_normalization_4/ReadVariableOp-^model/batch_normalization_4/ReadVariableOp_1#^model/dense/BiasAdd/ReadVariableOp"^model/dense/MatMul/ReadVariableOp%^model/dense_1/BiasAdd/ReadVariableOp$^model/dense_1/MatMul/ReadVariableOp^model/stream/AssignVariableOp#^model/stream/concat/ReadVariableOp+^model/stream/conv2d/BiasAdd/ReadVariableOp*^model/stream/conv2d/Conv2D/ReadVariableOp ^model/stream_1/AssignVariableOp%^model/stream_1/concat/ReadVariableOp7^model/stream_1/depthwise_conv2d/BiasAdd/ReadVariableOp9^model/stream_1/depthwise_conv2d/depthwise/ReadVariableOp ^model/stream_2/AssignVariableOp%^model/stream_2/concat/ReadVariableOp9^model/stream_2/depthwise_conv2d_1/BiasAdd/ReadVariableOp;^model/stream_2/depthwise_conv2d_1/depthwise/ReadVariableOp ^model/stream_3/AssignVariableOp%^model/stream_3/concat/ReadVariableOp ^model/stream_4/AssignVariableOp%^model/stream_4/concat/ReadVariableOp9^model/stream_4/depthwise_conv2d_2/BiasAdd/ReadVariableOp;^model/stream_4/depthwise_conv2d_2/depthwise/ReadVariableOp ^model/stream_5/AssignVariableOp%^model/stream_5/concat/ReadVariableOp9^model/stream_5/depthwise_conv2d_3/BiasAdd/ReadVariableOp;^model/stream_5/depthwise_conv2d_3/depthwise/ReadVariableOp ^model/stream_6/AssignVariableOp^model/stream_6/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*x
_input_shapesg
e:�: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2z
;model/batch_normalization/FusedBatchNormV3/ReadVariableOp_1;model/batch_normalization/FusedBatchNormV3/ReadVariableOp_12v
9model/batch_normalization/FusedBatchNormV3/ReadVariableOp9model/batch_normalization/FusedBatchNormV3/ReadVariableOp2X
*model/batch_normalization/ReadVariableOp_1*model/batch_normalization/ReadVariableOp_12T
(model/batch_normalization/ReadVariableOp(model/batch_normalization/ReadVariableOp2~
=model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1=model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12z
;model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp;model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2\
,model/batch_normalization_1/ReadVariableOp_1,model/batch_normalization_1/ReadVariableOp_12X
*model/batch_normalization_1/ReadVariableOp*model/batch_normalization_1/ReadVariableOp2~
=model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1=model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12z
;model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp;model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp2\
,model/batch_normalization_2/ReadVariableOp_1,model/batch_normalization_2/ReadVariableOp_12X
*model/batch_normalization_2/ReadVariableOp*model/batch_normalization_2/ReadVariableOp2~
=model/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1=model/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12z
;model/batch_normalization_3/FusedBatchNormV3/ReadVariableOp;model/batch_normalization_3/FusedBatchNormV3/ReadVariableOp2\
,model/batch_normalization_3/ReadVariableOp_1,model/batch_normalization_3/ReadVariableOp_12X
*model/batch_normalization_3/ReadVariableOp*model/batch_normalization_3/ReadVariableOp2~
=model/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1=model/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12z
;model/batch_normalization_4/FusedBatchNormV3/ReadVariableOp;model/batch_normalization_4/FusedBatchNormV3/ReadVariableOp2\
,model/batch_normalization_4/ReadVariableOp_1,model/batch_normalization_4/ReadVariableOp_12X
*model/batch_normalization_4/ReadVariableOp*model/batch_normalization_4/ReadVariableOp2H
"model/dense/BiasAdd/ReadVariableOp"model/dense/BiasAdd/ReadVariableOp2F
!model/dense/MatMul/ReadVariableOp!model/dense/MatMul/ReadVariableOp2L
$model/dense_1/BiasAdd/ReadVariableOp$model/dense_1/BiasAdd/ReadVariableOp2J
#model/dense_1/MatMul/ReadVariableOp#model/dense_1/MatMul/ReadVariableOp2>
model/stream/AssignVariableOpmodel/stream/AssignVariableOp2H
"model/stream/concat/ReadVariableOp"model/stream/concat/ReadVariableOp2X
*model/stream/conv2d/BiasAdd/ReadVariableOp*model/stream/conv2d/BiasAdd/ReadVariableOp2V
)model/stream/conv2d/Conv2D/ReadVariableOp)model/stream/conv2d/Conv2D/ReadVariableOp2B
model/stream_1/AssignVariableOpmodel/stream_1/AssignVariableOp2L
$model/stream_1/concat/ReadVariableOp$model/stream_1/concat/ReadVariableOp2p
6model/stream_1/depthwise_conv2d/BiasAdd/ReadVariableOp6model/stream_1/depthwise_conv2d/BiasAdd/ReadVariableOp2t
8model/stream_1/depthwise_conv2d/depthwise/ReadVariableOp8model/stream_1/depthwise_conv2d/depthwise/ReadVariableOp2B
model/stream_2/AssignVariableOpmodel/stream_2/AssignVariableOp2L
$model/stream_2/concat/ReadVariableOp$model/stream_2/concat/ReadVariableOp2t
8model/stream_2/depthwise_conv2d_1/BiasAdd/ReadVariableOp8model/stream_2/depthwise_conv2d_1/BiasAdd/ReadVariableOp2x
:model/stream_2/depthwise_conv2d_1/depthwise/ReadVariableOp:model/stream_2/depthwise_conv2d_1/depthwise/ReadVariableOp2B
model/stream_3/AssignVariableOpmodel/stream_3/AssignVariableOp2L
$model/stream_3/concat/ReadVariableOp$model/stream_3/concat/ReadVariableOp2B
model/stream_4/AssignVariableOpmodel/stream_4/AssignVariableOp2L
$model/stream_4/concat/ReadVariableOp$model/stream_4/concat/ReadVariableOp2t
8model/stream_4/depthwise_conv2d_2/BiasAdd/ReadVariableOp8model/stream_4/depthwise_conv2d_2/BiasAdd/ReadVariableOp2x
:model/stream_4/depthwise_conv2d_2/depthwise/ReadVariableOp:model/stream_4/depthwise_conv2d_2/depthwise/ReadVariableOp2B
model/stream_5/AssignVariableOpmodel/stream_5/AssignVariableOp2L
$model/stream_5/concat/ReadVariableOp$model/stream_5/concat/ReadVariableOp2t
8model/stream_5/depthwise_conv2d_3/BiasAdd/ReadVariableOp8model/stream_5/depthwise_conv2d_3/BiasAdd/ReadVariableOp2x
:model/stream_5/depthwise_conv2d_3/depthwise/ReadVariableOp:model/stream_5/depthwise_conv2d_3/depthwise/ReadVariableOp2B
model/stream_6/AssignVariableOpmodel/stream_6/AssignVariableOp2>
model/stream_6/ReadVariableOpmodel/stream_6/ReadVariableOp:S O
'
_output_shapes
:�
$
_user_specified_name
IEGM_input
�
�
C__inference_stream_2_layer_call_and_return_conditional_losses_12544

inputsI
/concat_readvariableop_streaming_stream_2_states:_
Edepthwise_conv2d_1_depthwise_readvariableop_stream_2_depthwise_kernel:E
7depthwise_conv2d_1_biasadd_readvariableop_stream_2_bias:
identity��AssignVariableOp�concat/ReadVariableOp�)depthwise_conv2d_1/BiasAdd/ReadVariableOp�+depthwise_conv2d_1/depthwise/ReadVariableOp�
concat/ReadVariableOpReadVariableOp/concat_readvariableop_streaming_stream_2_states*&
_output_shapes
:*
dtype0M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatConcatV2concat/ReadVariableOp:value:0inputsconcat/axis:output:0*
N*
T0*&
_output_shapes
:h
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ����    j
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            j
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         �
strided_sliceStridedSliceconcat:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*&
_output_shapes
:*

begin_mask*
end_mask�
AssignVariableOpAssignVariableOp/concat_readvariableop_streaming_stream_2_statesstrided_slice:output:0^concat/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
+depthwise_conv2d_1/depthwise/ReadVariableOpReadVariableOpEdepthwise_conv2d_1_depthwise_readvariableop_stream_2_depthwise_kernel^AssignVariableOp*&
_output_shapes
:*
dtype0�
"depthwise_conv2d_1/depthwise/ShapeConst^AssignVariableOp*
_output_shapes
:*
dtype0*%
valueB"            �
*depthwise_conv2d_1/depthwise/dilation_rateConst^AssignVariableOp*
_output_shapes
:*
dtype0*
valueB"      �
depthwise_conv2d_1/depthwiseDepthwiseConv2dNativeconcat:output:03depthwise_conv2d_1/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
:*
paddingVALID*
strides
�
)depthwise_conv2d_1/BiasAdd/ReadVariableOpReadVariableOp7depthwise_conv2d_1_biasadd_readvariableop_stream_2_bias^AssignVariableOp*
_output_shapes
:*
dtype0�
depthwise_conv2d_1/BiasAddBiasAdd%depthwise_conv2d_1/depthwise:output:01depthwise_conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:q
IdentityIdentity#depthwise_conv2d_1/BiasAdd:output:0^NoOp*
T0*&
_output_shapes
:�
NoOpNoOp^AssignVariableOp^concat/ReadVariableOp*^depthwise_conv2d_1/BiasAdd/ReadVariableOp,^depthwise_conv2d_1/depthwise/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*+
_input_shapes
:: : : 2$
AssignVariableOpAssignVariableOp2.
concat/ReadVariableOpconcat/ReadVariableOp2V
)depthwise_conv2d_1/BiasAdd/ReadVariableOp)depthwise_conv2d_1/BiasAdd/ReadVariableOp2Z
+depthwise_conv2d_1/depthwise/ReadVariableOp+depthwise_conv2d_1/depthwise/ReadVariableOp:N J
&
_output_shapes
:
 
_user_specified_nameinputs
��
�.
@__inference_model_layer_call_and_return_conditional_losses_12324

inputsN
4stream_concat_readvariableop_streaming_stream_states:K
1stream_conv2d_conv2d_readvariableop_stream_kernel:>
0stream_conv2d_biasadd_readvariableop_stream_bias:T
Fbatch_normalization_readvariableop_streaming_batch_normalization_gamma:U
Gbatch_normalization_readvariableop_1_streaming_batch_normalization_beta:k
]batch_normalization_fusedbatchnormv3_readvariableop_streaming_batch_normalization_moving_mean:q
cbatch_normalization_fusedbatchnormv3_readvariableop_1_streaming_batch_normalization_moving_variance:R
8stream_1_concat_readvariableop_streaming_stream_1_states:f
Lstream_1_depthwise_conv2d_depthwise_readvariableop_stream_1_depthwise_kernel:L
>stream_1_depthwise_conv2d_biasadd_readvariableop_stream_1_bias:X
Jbatch_normalization_1_readvariableop_streaming_batch_normalization_1_gamma:Y
Kbatch_normalization_1_readvariableop_1_streaming_batch_normalization_1_beta:o
abatch_normalization_1_fusedbatchnormv3_readvariableop_streaming_batch_normalization_1_moving_mean:u
gbatch_normalization_1_fusedbatchnormv3_readvariableop_1_streaming_batch_normalization_1_moving_variance:R
8stream_2_concat_readvariableop_streaming_stream_2_states:h
Nstream_2_depthwise_conv2d_1_depthwise_readvariableop_stream_2_depthwise_kernel:N
@stream_2_depthwise_conv2d_1_biasadd_readvariableop_stream_2_bias:X
Jbatch_normalization_2_readvariableop_streaming_batch_normalization_2_gamma:Y
Kbatch_normalization_2_readvariableop_1_streaming_batch_normalization_2_beta:o
abatch_normalization_2_fusedbatchnormv3_readvariableop_streaming_batch_normalization_2_moving_mean:u
gbatch_normalization_2_fusedbatchnormv3_readvariableop_1_streaming_batch_normalization_2_moving_variance:R
8stream_3_concat_readvariableop_streaming_stream_3_states:R
8stream_4_concat_readvariableop_streaming_stream_4_states:h
Nstream_4_depthwise_conv2d_2_depthwise_readvariableop_stream_4_depthwise_kernel:N
@stream_4_depthwise_conv2d_2_biasadd_readvariableop_stream_4_bias:X
Jbatch_normalization_3_readvariableop_streaming_batch_normalization_3_gamma:Y
Kbatch_normalization_3_readvariableop_1_streaming_batch_normalization_3_beta:o
abatch_normalization_3_fusedbatchnormv3_readvariableop_streaming_batch_normalization_3_moving_mean:u
gbatch_normalization_3_fusedbatchnormv3_readvariableop_1_streaming_batch_normalization_3_moving_variance:R
8stream_5_concat_readvariableop_streaming_stream_5_states:h
Nstream_5_depthwise_conv2d_3_depthwise_readvariableop_stream_5_depthwise_kernel:N
@stream_5_depthwise_conv2d_3_biasadd_readvariableop_stream_5_bias:X
Jbatch_normalization_4_readvariableop_streaming_batch_normalization_4_gamma:Y
Kbatch_normalization_4_readvariableop_1_streaming_batch_normalization_4_beta:o
abatch_normalization_4_fusedbatchnormv3_readvariableop_streaming_batch_normalization_4_moving_mean:u
gbatch_normalization_4_fusedbatchnormv3_readvariableop_1_streaming_batch_normalization_4_moving_variance:K
1stream_6_readvariableop_streaming_stream_6_states:D
2dense_matmul_readvariableop_streaming_dense_kernel:0
?
1dense_biasadd_readvariableop_streaming_dense_bias:
H
6dense_1_matmul_readvariableop_streaming_dense_1_kernel:
C
5dense_1_biasadd_readvariableop_streaming_dense_1_bias:
identity��3batch_normalization/FusedBatchNormV3/ReadVariableOp�5batch_normalization/FusedBatchNormV3/ReadVariableOp_1�"batch_normalization/ReadVariableOp�$batch_normalization/ReadVariableOp_1�5batch_normalization_1/FusedBatchNormV3/ReadVariableOp�7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_1/ReadVariableOp�&batch_normalization_1/ReadVariableOp_1�5batch_normalization_2/FusedBatchNormV3/ReadVariableOp�7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_2/ReadVariableOp�&batch_normalization_2/ReadVariableOp_1�5batch_normalization_3/FusedBatchNormV3/ReadVariableOp�7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_3/ReadVariableOp�&batch_normalization_3/ReadVariableOp_1�5batch_normalization_4/FusedBatchNormV3/ReadVariableOp�7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_4/ReadVariableOp�&batch_normalization_4/ReadVariableOp_1�dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�stream/AssignVariableOp�stream/concat/ReadVariableOp�$stream/conv2d/BiasAdd/ReadVariableOp�#stream/conv2d/Conv2D/ReadVariableOp�stream_1/AssignVariableOp�stream_1/concat/ReadVariableOp�0stream_1/depthwise_conv2d/BiasAdd/ReadVariableOp�2stream_1/depthwise_conv2d/depthwise/ReadVariableOp�stream_2/AssignVariableOp�stream_2/concat/ReadVariableOp�2stream_2/depthwise_conv2d_1/BiasAdd/ReadVariableOp�4stream_2/depthwise_conv2d_1/depthwise/ReadVariableOp�stream_3/AssignVariableOp�stream_3/concat/ReadVariableOp�stream_4/AssignVariableOp�stream_4/concat/ReadVariableOp�2stream_4/depthwise_conv2d_2/BiasAdd/ReadVariableOp�4stream_4/depthwise_conv2d_2/depthwise/ReadVariableOp�stream_5/AssignVariableOp�stream_5/concat/ReadVariableOp�2stream_5/depthwise_conv2d_3/BiasAdd/ReadVariableOp�4stream_5/depthwise_conv2d_3/depthwise/ReadVariableOp�stream_6/AssignVariableOp�stream_6/ReadVariableOp�
max_pooling2d/MaxPoolMaxPoolinputs*&
_output_shapes
:H*
ksize
*
paddingVALID*
strides
�
stream/concat/ReadVariableOpReadVariableOp4stream_concat_readvariableop_streaming_stream_states*&
_output_shapes
:*
dtype0T
stream/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
stream/concatConcatV2$stream/concat/ReadVariableOp:value:0max_pooling2d/MaxPool:output:0stream/concat/axis:output:0*
N*
T0*&
_output_shapes
:Lo
stream/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ����    q
stream/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            q
stream/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         �
stream/strided_sliceStridedSlicestream/concat:output:0#stream/strided_slice/stack:output:0%stream/strided_slice/stack_1:output:0%stream/strided_slice/stack_2:output:0*
Index0*
T0*&
_output_shapes
:*

begin_mask*
end_mask�
stream/AssignVariableOpAssignVariableOp4stream_concat_readvariableop_streaming_stream_statesstream/strided_slice:output:0^stream/concat/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
#stream/conv2d/Conv2D/ReadVariableOpReadVariableOp1stream_conv2d_conv2d_readvariableop_stream_kernel^stream/AssignVariableOp*&
_output_shapes
:*
dtype0�
stream/conv2d/Conv2DConv2Dstream/concat:output:0+stream/conv2d/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:$*
paddingVALID*
strides
�
$stream/conv2d/BiasAdd/ReadVariableOpReadVariableOp0stream_conv2d_biasadd_readvariableop_stream_bias^stream/AssignVariableOp*
_output_shapes
:*
dtype0�
stream/conv2d/BiasAddBiasAddstream/conv2d/Conv2D:output:0,stream/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:$�
"batch_normalization/ReadVariableOpReadVariableOpFbatch_normalization_readvariableop_streaming_batch_normalization_gamma*
_output_shapes
:*
dtype0�
$batch_normalization/ReadVariableOp_1ReadVariableOpGbatch_normalization_readvariableop_1_streaming_batch_normalization_beta*
_output_shapes
:*
dtype0�
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp]batch_normalization_fusedbatchnormv3_readvariableop_streaming_batch_normalization_moving_mean*
_output_shapes
:*
dtype0�
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpcbatch_normalization_fusedbatchnormv3_readvariableop_1_streaming_batch_normalization_moving_variance*
_output_shapes
:*
dtype0�
$batch_normalization/FusedBatchNormV3FusedBatchNormV3stream/conv2d/BiasAdd:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:$:::::*
epsilon%��'7*
is_training( r
activation/ReluRelu(batch_normalization/FusedBatchNormV3:y:0*
T0*&
_output_shapes
:$�
stream_1/concat/ReadVariableOpReadVariableOp8stream_1_concat_readvariableop_streaming_stream_1_states*&
_output_shapes
:*
dtype0V
stream_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
stream_1/concatConcatV2&stream_1/concat/ReadVariableOp:value:0activation/Relu:activations:0stream_1/concat/axis:output:0*
N*
T0*&
_output_shapes
:'q
stream_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ����    s
stream_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            s
stream_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         �
stream_1/strided_sliceStridedSlicestream_1/concat:output:0%stream_1/strided_slice/stack:output:0'stream_1/strided_slice/stack_1:output:0'stream_1/strided_slice/stack_2:output:0*
Index0*
T0*&
_output_shapes
:*

begin_mask*
end_mask�
stream_1/AssignVariableOpAssignVariableOp8stream_1_concat_readvariableop_streaming_stream_1_statesstream_1/strided_slice:output:0^stream_1/concat/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
2stream_1/depthwise_conv2d/depthwise/ReadVariableOpReadVariableOpLstream_1_depthwise_conv2d_depthwise_readvariableop_stream_1_depthwise_kernel^stream_1/AssignVariableOp*&
_output_shapes
:*
dtype0�
)stream_1/depthwise_conv2d/depthwise/ShapeConst^stream_1/AssignVariableOp*
_output_shapes
:*
dtype0*%
valueB"            �
1stream_1/depthwise_conv2d/depthwise/dilation_rateConst^stream_1/AssignVariableOp*
_output_shapes
:*
dtype0*
valueB"      �
#stream_1/depthwise_conv2d/depthwiseDepthwiseConv2dNativestream_1/concat:output:0:stream_1/depthwise_conv2d/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
:*
paddingVALID*
strides
�
0stream_1/depthwise_conv2d/BiasAdd/ReadVariableOpReadVariableOp>stream_1_depthwise_conv2d_biasadd_readvariableop_stream_1_bias^stream_1/AssignVariableOp*
_output_shapes
:*
dtype0�
!stream_1/depthwise_conv2d/BiasAddBiasAdd,stream_1/depthwise_conv2d/depthwise:output:08stream_1/depthwise_conv2d/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:�
$batch_normalization_1/ReadVariableOpReadVariableOpJbatch_normalization_1_readvariableop_streaming_batch_normalization_1_gamma*
_output_shapes
:*
dtype0�
&batch_normalization_1/ReadVariableOp_1ReadVariableOpKbatch_normalization_1_readvariableop_1_streaming_batch_normalization_1_beta*
_output_shapes
:*
dtype0�
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpabatch_normalization_1_fusedbatchnormv3_readvariableop_streaming_batch_normalization_1_moving_mean*
_output_shapes
:*
dtype0�
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpgbatch_normalization_1_fusedbatchnormv3_readvariableop_1_streaming_batch_normalization_1_moving_variance*
_output_shapes
:*
dtype0�
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3*stream_1/depthwise_conv2d/BiasAdd:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.::::::*
epsilon%��'7*
is_training( v
activation_1/ReluRelu*batch_normalization_1/FusedBatchNormV3:y:0*
T0*&
_output_shapes
:�
stream_2/concat/ReadVariableOpReadVariableOp8stream_2_concat_readvariableop_streaming_stream_2_states*&
_output_shapes
:*
dtype0V
stream_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
stream_2/concatConcatV2&stream_2/concat/ReadVariableOp:value:0activation_1/Relu:activations:0stream_2/concat/axis:output:0*
N*
T0*&
_output_shapes
:q
stream_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ����    s
stream_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            s
stream_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         �
stream_2/strided_sliceStridedSlicestream_2/concat:output:0%stream_2/strided_slice/stack:output:0'stream_2/strided_slice/stack_1:output:0'stream_2/strided_slice/stack_2:output:0*
Index0*
T0*&
_output_shapes
:*

begin_mask*
end_mask�
stream_2/AssignVariableOpAssignVariableOp8stream_2_concat_readvariableop_streaming_stream_2_statesstream_2/strided_slice:output:0^stream_2/concat/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
4stream_2/depthwise_conv2d_1/depthwise/ReadVariableOpReadVariableOpNstream_2_depthwise_conv2d_1_depthwise_readvariableop_stream_2_depthwise_kernel^stream_2/AssignVariableOp*&
_output_shapes
:*
dtype0�
+stream_2/depthwise_conv2d_1/depthwise/ShapeConst^stream_2/AssignVariableOp*
_output_shapes
:*
dtype0*%
valueB"            �
3stream_2/depthwise_conv2d_1/depthwise/dilation_rateConst^stream_2/AssignVariableOp*
_output_shapes
:*
dtype0*
valueB"      �
%stream_2/depthwise_conv2d_1/depthwiseDepthwiseConv2dNativestream_2/concat:output:0<stream_2/depthwise_conv2d_1/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
:*
paddingVALID*
strides
�
2stream_2/depthwise_conv2d_1/BiasAdd/ReadVariableOpReadVariableOp@stream_2_depthwise_conv2d_1_biasadd_readvariableop_stream_2_bias^stream_2/AssignVariableOp*
_output_shapes
:*
dtype0�
#stream_2/depthwise_conv2d_1/BiasAddBiasAdd.stream_2/depthwise_conv2d_1/depthwise:output:0:stream_2/depthwise_conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:�
$batch_normalization_2/ReadVariableOpReadVariableOpJbatch_normalization_2_readvariableop_streaming_batch_normalization_2_gamma*
_output_shapes
:*
dtype0�
&batch_normalization_2/ReadVariableOp_1ReadVariableOpKbatch_normalization_2_readvariableop_1_streaming_batch_normalization_2_beta*
_output_shapes
:*
dtype0�
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpabatch_normalization_2_fusedbatchnormv3_readvariableop_streaming_batch_normalization_2_moving_mean*
_output_shapes
:*
dtype0�
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpgbatch_normalization_2_fusedbatchnormv3_readvariableop_1_streaming_batch_normalization_2_moving_variance*
_output_shapes
:*
dtype0�
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3,stream_2/depthwise_conv2d_1/BiasAdd:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.::::::*
epsilon%��'7*
is_training( v
activation_2/ReluRelu*batch_normalization_2/FusedBatchNormV3:y:0*
T0*&
_output_shapes
:�
stream_3/concat/ReadVariableOpReadVariableOp8stream_3_concat_readvariableop_streaming_stream_3_states*&
_output_shapes
:*
dtype0V
stream_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
stream_3/concatConcatV2&stream_3/concat/ReadVariableOp:value:0activation_2/Relu:activations:0stream_3/concat/axis:output:0*
N*
T0*&
_output_shapes
:q
stream_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ����    s
stream_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            s
stream_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         �
stream_3/strided_sliceStridedSlicestream_3/concat:output:0%stream_3/strided_slice/stack:output:0'stream_3/strided_slice/stack_1:output:0'stream_3/strided_slice/stack_2:output:0*
Index0*
T0*&
_output_shapes
:*

begin_mask*
end_mask�
stream_3/AssignVariableOpAssignVariableOp8stream_3_concat_readvariableop_streaming_stream_3_statesstream_3/strided_slice:output:0^stream_3/concat/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
"stream_3/average_pooling2d/AvgPoolAvgPoolstream_3/concat:output:0^stream_3/AssignVariableOp*
T0*&
_output_shapes
:*
ksize
*
paddingVALID*
strides
�
stream_4/concat/ReadVariableOpReadVariableOp8stream_4_concat_readvariableop_streaming_stream_4_states*&
_output_shapes
:*
dtype0V
stream_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
stream_4/concatConcatV2&stream_4/concat/ReadVariableOp:value:0+stream_3/average_pooling2d/AvgPool:output:0stream_4/concat/axis:output:0*
N*
T0*&
_output_shapes
:	q
stream_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ����    s
stream_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            s
stream_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         �
stream_4/strided_sliceStridedSlicestream_4/concat:output:0%stream_4/strided_slice/stack:output:0'stream_4/strided_slice/stack_1:output:0'stream_4/strided_slice/stack_2:output:0*
Index0*
T0*&
_output_shapes
:*

begin_mask*
end_mask�
stream_4/AssignVariableOpAssignVariableOp8stream_4_concat_readvariableop_streaming_stream_4_statesstream_4/strided_slice:output:0^stream_4/concat/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
4stream_4/depthwise_conv2d_2/depthwise/ReadVariableOpReadVariableOpNstream_4_depthwise_conv2d_2_depthwise_readvariableop_stream_4_depthwise_kernel^stream_4/AssignVariableOp*&
_output_shapes
:*
dtype0�
+stream_4/depthwise_conv2d_2/depthwise/ShapeConst^stream_4/AssignVariableOp*
_output_shapes
:*
dtype0*%
valueB"            �
3stream_4/depthwise_conv2d_2/depthwise/dilation_rateConst^stream_4/AssignVariableOp*
_output_shapes
:*
dtype0*
valueB"      �
%stream_4/depthwise_conv2d_2/depthwiseDepthwiseConv2dNativestream_4/concat:output:0<stream_4/depthwise_conv2d_2/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
:*
paddingVALID*
strides
�
2stream_4/depthwise_conv2d_2/BiasAdd/ReadVariableOpReadVariableOp@stream_4_depthwise_conv2d_2_biasadd_readvariableop_stream_4_bias^stream_4/AssignVariableOp*
_output_shapes
:*
dtype0�
#stream_4/depthwise_conv2d_2/BiasAddBiasAdd.stream_4/depthwise_conv2d_2/depthwise:output:0:stream_4/depthwise_conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:�
$batch_normalization_3/ReadVariableOpReadVariableOpJbatch_normalization_3_readvariableop_streaming_batch_normalization_3_gamma*
_output_shapes
:*
dtype0�
&batch_normalization_3/ReadVariableOp_1ReadVariableOpKbatch_normalization_3_readvariableop_1_streaming_batch_normalization_3_beta*
_output_shapes
:*
dtype0�
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpabatch_normalization_3_fusedbatchnormv3_readvariableop_streaming_batch_normalization_3_moving_mean*
_output_shapes
:*
dtype0�
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpgbatch_normalization_3_fusedbatchnormv3_readvariableop_1_streaming_batch_normalization_3_moving_variance*
_output_shapes
:*
dtype0�
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3,stream_4/depthwise_conv2d_2/BiasAdd:output:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.::::::*
epsilon%��'7*
is_training( v
activation_3/ReluRelu*batch_normalization_3/FusedBatchNormV3:y:0*
T0*&
_output_shapes
:�
stream_5/concat/ReadVariableOpReadVariableOp8stream_5_concat_readvariableop_streaming_stream_5_states*&
_output_shapes
:*
dtype0V
stream_5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
stream_5/concatConcatV2&stream_5/concat/ReadVariableOp:value:0activation_3/Relu:activations:0stream_5/concat/axis:output:0*
N*
T0*&
_output_shapes
:q
stream_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ����    s
stream_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            s
stream_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         �
stream_5/strided_sliceStridedSlicestream_5/concat:output:0%stream_5/strided_slice/stack:output:0'stream_5/strided_slice/stack_1:output:0'stream_5/strided_slice/stack_2:output:0*
Index0*
T0*&
_output_shapes
:*

begin_mask*
end_mask�
stream_5/AssignVariableOpAssignVariableOp8stream_5_concat_readvariableop_streaming_stream_5_statesstream_5/strided_slice:output:0^stream_5/concat/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
4stream_5/depthwise_conv2d_3/depthwise/ReadVariableOpReadVariableOpNstream_5_depthwise_conv2d_3_depthwise_readvariableop_stream_5_depthwise_kernel^stream_5/AssignVariableOp*&
_output_shapes
:*
dtype0�
+stream_5/depthwise_conv2d_3/depthwise/ShapeConst^stream_5/AssignVariableOp*
_output_shapes
:*
dtype0*%
valueB"            �
3stream_5/depthwise_conv2d_3/depthwise/dilation_rateConst^stream_5/AssignVariableOp*
_output_shapes
:*
dtype0*
valueB"      �
%stream_5/depthwise_conv2d_3/depthwiseDepthwiseConv2dNativestream_5/concat:output:0<stream_5/depthwise_conv2d_3/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
:*
paddingVALID*
strides
�
2stream_5/depthwise_conv2d_3/BiasAdd/ReadVariableOpReadVariableOp@stream_5_depthwise_conv2d_3_biasadd_readvariableop_stream_5_bias^stream_5/AssignVariableOp*
_output_shapes
:*
dtype0�
#stream_5/depthwise_conv2d_3/BiasAddBiasAdd.stream_5/depthwise_conv2d_3/depthwise:output:0:stream_5/depthwise_conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:�
$batch_normalization_4/ReadVariableOpReadVariableOpJbatch_normalization_4_readvariableop_streaming_batch_normalization_4_gamma*
_output_shapes
:*
dtype0�
&batch_normalization_4/ReadVariableOp_1ReadVariableOpKbatch_normalization_4_readvariableop_1_streaming_batch_normalization_4_beta*
_output_shapes
:*
dtype0�
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpabatch_normalization_4_fusedbatchnormv3_readvariableop_streaming_batch_normalization_4_moving_mean*
_output_shapes
:*
dtype0�
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpgbatch_normalization_4_fusedbatchnormv3_readvariableop_1_streaming_batch_normalization_4_moving_variance*
_output_shapes
:*
dtype0�
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3,stream_5/depthwise_conv2d_3/BiasAdd:output:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.::::::*
epsilon%��'7*
is_training( v
activation_4/ReluRelu*batch_normalization_4/FusedBatchNormV3:y:0*
T0*&
_output_shapes
:n
dropout/IdentityIdentityactivation_4/Relu:activations:0*
T0*&
_output_shapes
:�
stream_6/ReadVariableOpReadVariableOp1stream_6_readvariableop_streaming_stream_6_states*&
_output_shapes
:*
dtype0q
stream_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           s
stream_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           s
stream_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         �
stream_6/strided_sliceStridedSlicestream_6/ReadVariableOp:value:0%stream_6/strided_slice/stack:output:0'stream_6/strided_slice/stack_1:output:0'stream_6/strided_slice/stack_2:output:0*
Index0*
T0*&
_output_shapes
:*

begin_mask*
end_maskV
stream_6/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
stream_6/concatConcatV2stream_6/strided_slice:output:0dropout/Identity:output:0stream_6/concat/axis:output:0*
N*
T0*&
_output_shapes
:�
stream_6/AssignVariableOpAssignVariableOp1stream_6_readvariableop_streaming_stream_6_statesstream_6/concat:output:0^stream_6/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
stream_6/flatten/ConstConst^stream_6/AssignVariableOp*
_output_shapes
:*
dtype0*
valueB"����0   �
stream_6/flatten/ReshapeReshapestream_6/concat:output:0stream_6/flatten/Const:output:0*
T0*
_output_shapes

:0�
dense/MatMul/ReadVariableOpReadVariableOp2dense_matmul_readvariableop_streaming_dense_kernel*
_output_shapes

:0
*
dtype0�
dense/MatMulMatMul!stream_6/flatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:
�
dense/BiasAdd/ReadVariableOpReadVariableOp1dense_biasadd_readvariableop_streaming_dense_bias*
_output_shapes
:
*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:
S

dense/ReluReludense/BiasAdd:output:0*
T0*
_output_shapes

:
�
dense_1/MatMul/ReadVariableOpReadVariableOp6dense_1_matmul_readvariableop_streaming_dense_1_kernel*
_output_shapes

:
*
dtype0�
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:�
dense_1/BiasAdd/ReadVariableOpReadVariableOp5dense_1_biasadd_readvariableop_streaming_dense_1_bias*
_output_shapes
:*
dtype0�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:]
dense_1/SoftmaxSoftmaxdense_1/BiasAdd:output:0*
T0*
_output_shapes

:_
IdentityIdentitydense_1/Softmax:softmax:0^NoOp*
T0*
_output_shapes

:�
NoOpNoOp4^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_16^batch_normalization_2/FusedBatchNormV3/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_16^batch_normalization_3/FusedBatchNormV3/ReadVariableOp8^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_3/ReadVariableOp'^batch_normalization_3/ReadVariableOp_16^batch_normalization_4/FusedBatchNormV3/ReadVariableOp8^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_4/ReadVariableOp'^batch_normalization_4/ReadVariableOp_1^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^stream/AssignVariableOp^stream/concat/ReadVariableOp%^stream/conv2d/BiasAdd/ReadVariableOp$^stream/conv2d/Conv2D/ReadVariableOp^stream_1/AssignVariableOp^stream_1/concat/ReadVariableOp1^stream_1/depthwise_conv2d/BiasAdd/ReadVariableOp3^stream_1/depthwise_conv2d/depthwise/ReadVariableOp^stream_2/AssignVariableOp^stream_2/concat/ReadVariableOp3^stream_2/depthwise_conv2d_1/BiasAdd/ReadVariableOp5^stream_2/depthwise_conv2d_1/depthwise/ReadVariableOp^stream_3/AssignVariableOp^stream_3/concat/ReadVariableOp^stream_4/AssignVariableOp^stream_4/concat/ReadVariableOp3^stream_4/depthwise_conv2d_2/BiasAdd/ReadVariableOp5^stream_4/depthwise_conv2d_2/depthwise/ReadVariableOp^stream_5/AssignVariableOp^stream_5/concat/ReadVariableOp3^stream_5/depthwise_conv2d_3/BiasAdd/ReadVariableOp5^stream_5/depthwise_conv2d_3/depthwise/ReadVariableOp^stream_6/AssignVariableOp^stream_6/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*x
_input_shapesg
e:�: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2r
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_17batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12n
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_12L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2r
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_17batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12n
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp5batch_normalization_2/FusedBatchNormV3/ReadVariableOp2P
&batch_normalization_2/ReadVariableOp_1&batch_normalization_2/ReadVariableOp_12L
$batch_normalization_2/ReadVariableOp$batch_normalization_2/ReadVariableOp2r
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_17batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12n
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp5batch_normalization_3/FusedBatchNormV3/ReadVariableOp2P
&batch_normalization_3/ReadVariableOp_1&batch_normalization_3/ReadVariableOp_12L
$batch_normalization_3/ReadVariableOp$batch_normalization_3/ReadVariableOp2r
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_17batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12n
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp5batch_normalization_4/FusedBatchNormV3/ReadVariableOp2P
&batch_normalization_4/ReadVariableOp_1&batch_normalization_4/ReadVariableOp_12L
$batch_normalization_4/ReadVariableOp$batch_normalization_4/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp22
stream/AssignVariableOpstream/AssignVariableOp2<
stream/concat/ReadVariableOpstream/concat/ReadVariableOp2L
$stream/conv2d/BiasAdd/ReadVariableOp$stream/conv2d/BiasAdd/ReadVariableOp2J
#stream/conv2d/Conv2D/ReadVariableOp#stream/conv2d/Conv2D/ReadVariableOp26
stream_1/AssignVariableOpstream_1/AssignVariableOp2@
stream_1/concat/ReadVariableOpstream_1/concat/ReadVariableOp2d
0stream_1/depthwise_conv2d/BiasAdd/ReadVariableOp0stream_1/depthwise_conv2d/BiasAdd/ReadVariableOp2h
2stream_1/depthwise_conv2d/depthwise/ReadVariableOp2stream_1/depthwise_conv2d/depthwise/ReadVariableOp26
stream_2/AssignVariableOpstream_2/AssignVariableOp2@
stream_2/concat/ReadVariableOpstream_2/concat/ReadVariableOp2h
2stream_2/depthwise_conv2d_1/BiasAdd/ReadVariableOp2stream_2/depthwise_conv2d_1/BiasAdd/ReadVariableOp2l
4stream_2/depthwise_conv2d_1/depthwise/ReadVariableOp4stream_2/depthwise_conv2d_1/depthwise/ReadVariableOp26
stream_3/AssignVariableOpstream_3/AssignVariableOp2@
stream_3/concat/ReadVariableOpstream_3/concat/ReadVariableOp26
stream_4/AssignVariableOpstream_4/AssignVariableOp2@
stream_4/concat/ReadVariableOpstream_4/concat/ReadVariableOp2h
2stream_4/depthwise_conv2d_2/BiasAdd/ReadVariableOp2stream_4/depthwise_conv2d_2/BiasAdd/ReadVariableOp2l
4stream_4/depthwise_conv2d_2/depthwise/ReadVariableOp4stream_4/depthwise_conv2d_2/depthwise/ReadVariableOp26
stream_5/AssignVariableOpstream_5/AssignVariableOp2@
stream_5/concat/ReadVariableOpstream_5/concat/ReadVariableOp2h
2stream_5/depthwise_conv2d_3/BiasAdd/ReadVariableOp2stream_5/depthwise_conv2d_3/BiasAdd/ReadVariableOp2l
4stream_5/depthwise_conv2d_3/depthwise/ReadVariableOp4stream_5/depthwise_conv2d_3/depthwise/ReadVariableOp26
stream_6/AssignVariableOpstream_6/AssignVariableOp22
stream_6/ReadVariableOpstream_6/ReadVariableOp:O K
'
_output_shapes
:�
 
_user_specified_nameinputs
�
�
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_10668

inputsB
4readvariableop_streaming_batch_normalization_4_gamma:C
5readvariableop_1_streaming_batch_normalization_4_beta:Y
Kfusedbatchnormv3_readvariableop_streaming_batch_normalization_4_moving_mean:_
Qfusedbatchnormv3_readvariableop_1_streaming_batch_normalization_4_moving_variance:
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1
ReadVariableOpReadVariableOp4readvariableop_streaming_batch_normalization_4_gamma*
_output_shapes
:*
dtype0�
ReadVariableOp_1ReadVariableOp5readvariableop_1_streaming_batch_normalization_4_beta*
_output_shapes
:*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOpKfusedbatchnormv3_readvariableop_streaming_batch_normalization_4_moving_mean*
_output_shapes
:*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpQfusedbatchnormv3_readvariableop_1_streaming_batch_normalization_4_moving_variance*
_output_shapes
:*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%��'7*
exponential_avg_factor%fff?�
AssignNewValueAssignVariableOpKfusedbatchnormv3_readvariableop_streaming_batch_normalization_4_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOpQfusedbatchnormv3_readvariableop_1_streaming_batch_normalization_4_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
`
'__inference_dropout_layer_call_fn_12816

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_10918n
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*&
_output_shapes
:`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*%
_input_shapes
:22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
:
 
_user_specified_nameinputs
��
�)
__inference__traced_save_13173
file_prefixH
.read_disablecopyonread_streaming_stream_states:J
<read_1_disablecopyonread_streaming_batch_normalization_gamma:I
;read_2_disablecopyonread_streaming_batch_normalization_beta:P
Bread_3_disablecopyonread_streaming_batch_normalization_moving_mean:T
Fread_4_disablecopyonread_streaming_batch_normalization_moving_variance:L
2read_5_disablecopyonread_streaming_stream_1_states:L
>read_6_disablecopyonread_streaming_batch_normalization_1_gamma:K
=read_7_disablecopyonread_streaming_batch_normalization_1_beta:R
Dread_8_disablecopyonread_streaming_batch_normalization_1_moving_mean:V
Hread_9_disablecopyonread_streaming_batch_normalization_1_moving_variance:M
3read_10_disablecopyonread_streaming_stream_2_states:M
?read_11_disablecopyonread_streaming_batch_normalization_2_gamma:L
>read_12_disablecopyonread_streaming_batch_normalization_2_beta:S
Eread_13_disablecopyonread_streaming_batch_normalization_2_moving_mean:W
Iread_14_disablecopyonread_streaming_batch_normalization_2_moving_variance:M
3read_15_disablecopyonread_streaming_stream_3_states:M
3read_16_disablecopyonread_streaming_stream_4_states:M
?read_17_disablecopyonread_streaming_batch_normalization_3_gamma:L
>read_18_disablecopyonread_streaming_batch_normalization_3_beta:S
Eread_19_disablecopyonread_streaming_batch_normalization_3_moving_mean:W
Iread_20_disablecopyonread_streaming_batch_normalization_3_moving_variance:M
3read_21_disablecopyonread_streaming_stream_5_states:M
?read_22_disablecopyonread_streaming_batch_normalization_4_gamma:L
>read_23_disablecopyonread_streaming_batch_normalization_4_beta:S
Eread_24_disablecopyonread_streaming_batch_normalization_4_moving_mean:W
Iread_25_disablecopyonread_streaming_batch_normalization_4_moving_variance:M
3read_26_disablecopyonread_streaming_stream_6_states:B
0read_27_disablecopyonread_streaming_dense_kernel:0
<
.read_28_disablecopyonread_streaming_dense_bias:
D
2read_29_disablecopyonread_streaming_dense_1_kernel:
>
0read_30_disablecopyonread_streaming_dense_1_bias:A
'read_31_disablecopyonread_stream_kernel:3
%read_32_disablecopyonread_stream_bias:M
3read_33_disablecopyonread_stream_1_depthwise_kernel:5
'read_34_disablecopyonread_stream_1_bias:M
3read_35_disablecopyonread_stream_2_depthwise_kernel:5
'read_36_disablecopyonread_stream_2_bias:M
3read_37_disablecopyonread_stream_4_depthwise_kernel:5
'read_38_disablecopyonread_stream_4_bias:M
3read_39_disablecopyonread_stream_5_depthwise_kernel:5
'read_40_disablecopyonread_stream_5_bias:
savev2_const
identity_83��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_18/DisableCopyOnRead�Read_18/ReadVariableOp�Read_19/DisableCopyOnRead�Read_19/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_20/DisableCopyOnRead�Read_20/ReadVariableOp�Read_21/DisableCopyOnRead�Read_21/ReadVariableOp�Read_22/DisableCopyOnRead�Read_22/ReadVariableOp�Read_23/DisableCopyOnRead�Read_23/ReadVariableOp�Read_24/DisableCopyOnRead�Read_24/ReadVariableOp�Read_25/DisableCopyOnRead�Read_25/ReadVariableOp�Read_26/DisableCopyOnRead�Read_26/ReadVariableOp�Read_27/DisableCopyOnRead�Read_27/ReadVariableOp�Read_28/DisableCopyOnRead�Read_28/ReadVariableOp�Read_29/DisableCopyOnRead�Read_29/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_30/DisableCopyOnRead�Read_30/ReadVariableOp�Read_31/DisableCopyOnRead�Read_31/ReadVariableOp�Read_32/DisableCopyOnRead�Read_32/ReadVariableOp�Read_33/DisableCopyOnRead�Read_33/ReadVariableOp�Read_34/DisableCopyOnRead�Read_34/ReadVariableOp�Read_35/DisableCopyOnRead�Read_35/ReadVariableOp�Read_36/DisableCopyOnRead�Read_36/ReadVariableOp�Read_37/DisableCopyOnRead�Read_37/ReadVariableOp�Read_38/DisableCopyOnRead�Read_38/ReadVariableOp�Read_39/DisableCopyOnRead�Read_39/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_40/DisableCopyOnRead�Read_40/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
Read/DisableCopyOnReadDisableCopyOnRead.read_disablecopyonread_streaming_stream_states"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp.read_disablecopyonread_streaming_stream_states^Read/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0q
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:i

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*&
_output_shapes
:�
Read_1/DisableCopyOnReadDisableCopyOnRead<read_1_disablecopyonread_streaming_batch_normalization_gamma"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp<read_1_disablecopyonread_streaming_batch_normalization_gamma^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_2/DisableCopyOnReadDisableCopyOnRead;read_2_disablecopyonread_streaming_batch_normalization_beta"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp;read_2_disablecopyonread_streaming_batch_normalization_beta^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_3/DisableCopyOnReadDisableCopyOnReadBread_3_disablecopyonread_streaming_batch_normalization_moving_mean"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOpBread_3_disablecopyonread_streaming_batch_normalization_moving_mean^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_4/DisableCopyOnReadDisableCopyOnReadFread_4_disablecopyonread_streaming_batch_normalization_moving_variance"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOpFread_4_disablecopyonread_streaming_batch_normalization_moving_variance^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_5/DisableCopyOnReadDisableCopyOnRead2read_5_disablecopyonread_streaming_stream_1_states"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp2read_5_disablecopyonread_streaming_stream_1_states^Read_5/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0v
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:m
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*&
_output_shapes
:�
Read_6/DisableCopyOnReadDisableCopyOnRead>read_6_disablecopyonread_streaming_batch_normalization_1_gamma"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp>read_6_disablecopyonread_streaming_batch_normalization_1_gamma^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_7/DisableCopyOnReadDisableCopyOnRead=read_7_disablecopyonread_streaming_batch_normalization_1_beta"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp=read_7_disablecopyonread_streaming_batch_normalization_1_beta^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_8/DisableCopyOnReadDisableCopyOnReadDread_8_disablecopyonread_streaming_batch_normalization_1_moving_mean"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOpDread_8_disablecopyonread_streaming_batch_normalization_1_moving_mean^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_9/DisableCopyOnReadDisableCopyOnReadHread_9_disablecopyonread_streaming_batch_normalization_1_moving_variance"/device:CPU:0*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOpHread_9_disablecopyonread_streaming_batch_normalization_1_moving_variance^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_10/DisableCopyOnReadDisableCopyOnRead3read_10_disablecopyonread_streaming_stream_2_states"/device:CPU:0*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp3read_10_disablecopyonread_streaming_stream_2_states^Read_10/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0w
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:m
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*&
_output_shapes
:�
Read_11/DisableCopyOnReadDisableCopyOnRead?read_11_disablecopyonread_streaming_batch_normalization_2_gamma"/device:CPU:0*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp?read_11_disablecopyonread_streaming_batch_normalization_2_gamma^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_12/DisableCopyOnReadDisableCopyOnRead>read_12_disablecopyonread_streaming_batch_normalization_2_beta"/device:CPU:0*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOp>read_12_disablecopyonread_streaming_batch_normalization_2_beta^Read_12/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_13/DisableCopyOnReadDisableCopyOnReadEread_13_disablecopyonread_streaming_batch_normalization_2_moving_mean"/device:CPU:0*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOpEread_13_disablecopyonread_streaming_batch_normalization_2_moving_mean^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_14/DisableCopyOnReadDisableCopyOnReadIread_14_disablecopyonread_streaming_batch_normalization_2_moving_variance"/device:CPU:0*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOpIread_14_disablecopyonread_streaming_batch_normalization_2_moving_variance^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_15/DisableCopyOnReadDisableCopyOnRead3read_15_disablecopyonread_streaming_stream_3_states"/device:CPU:0*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOp3read_15_disablecopyonread_streaming_stream_3_states^Read_15/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0w
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:m
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*&
_output_shapes
:�
Read_16/DisableCopyOnReadDisableCopyOnRead3read_16_disablecopyonread_streaming_stream_4_states"/device:CPU:0*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOp3read_16_disablecopyonread_streaming_stream_4_states^Read_16/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0w
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:m
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*&
_output_shapes
:�
Read_17/DisableCopyOnReadDisableCopyOnRead?read_17_disablecopyonread_streaming_batch_normalization_3_gamma"/device:CPU:0*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOp?read_17_disablecopyonread_streaming_batch_normalization_3_gamma^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_18/DisableCopyOnReadDisableCopyOnRead>read_18_disablecopyonread_streaming_batch_normalization_3_beta"/device:CPU:0*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOp>read_18_disablecopyonread_streaming_batch_normalization_3_beta^Read_18/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_19/DisableCopyOnReadDisableCopyOnReadEread_19_disablecopyonread_streaming_batch_normalization_3_moving_mean"/device:CPU:0*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOpEread_19_disablecopyonread_streaming_batch_normalization_3_moving_mean^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_20/DisableCopyOnReadDisableCopyOnReadIread_20_disablecopyonread_streaming_batch_normalization_3_moving_variance"/device:CPU:0*
_output_shapes
 �
Read_20/ReadVariableOpReadVariableOpIread_20_disablecopyonread_streaming_batch_normalization_3_moving_variance^Read_20/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_21/DisableCopyOnReadDisableCopyOnRead3read_21_disablecopyonread_streaming_stream_5_states"/device:CPU:0*
_output_shapes
 �
Read_21/ReadVariableOpReadVariableOp3read_21_disablecopyonread_streaming_stream_5_states^Read_21/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0w
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:m
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*&
_output_shapes
:�
Read_22/DisableCopyOnReadDisableCopyOnRead?read_22_disablecopyonread_streaming_batch_normalization_4_gamma"/device:CPU:0*
_output_shapes
 �
Read_22/ReadVariableOpReadVariableOp?read_22_disablecopyonread_streaming_batch_normalization_4_gamma^Read_22/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_23/DisableCopyOnReadDisableCopyOnRead>read_23_disablecopyonread_streaming_batch_normalization_4_beta"/device:CPU:0*
_output_shapes
 �
Read_23/ReadVariableOpReadVariableOp>read_23_disablecopyonread_streaming_batch_normalization_4_beta^Read_23/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_24/DisableCopyOnReadDisableCopyOnReadEread_24_disablecopyonread_streaming_batch_normalization_4_moving_mean"/device:CPU:0*
_output_shapes
 �
Read_24/ReadVariableOpReadVariableOpEread_24_disablecopyonread_streaming_batch_normalization_4_moving_mean^Read_24/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_25/DisableCopyOnReadDisableCopyOnReadIread_25_disablecopyonread_streaming_batch_normalization_4_moving_variance"/device:CPU:0*
_output_shapes
 �
Read_25/ReadVariableOpReadVariableOpIread_25_disablecopyonread_streaming_batch_normalization_4_moving_variance^Read_25/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_26/DisableCopyOnReadDisableCopyOnRead3read_26_disablecopyonread_streaming_stream_6_states"/device:CPU:0*
_output_shapes
 �
Read_26/ReadVariableOpReadVariableOp3read_26_disablecopyonread_streaming_stream_6_states^Read_26/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0w
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:m
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0*&
_output_shapes
:�
Read_27/DisableCopyOnReadDisableCopyOnRead0read_27_disablecopyonread_streaming_dense_kernel"/device:CPU:0*
_output_shapes
 �
Read_27/ReadVariableOpReadVariableOp0read_27_disablecopyonread_streaming_dense_kernel^Read_27/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:0
*
dtype0o
Identity_54IdentityRead_27/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:0
e
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0*
_output_shapes

:0
�
Read_28/DisableCopyOnReadDisableCopyOnRead.read_28_disablecopyonread_streaming_dense_bias"/device:CPU:0*
_output_shapes
 �
Read_28/ReadVariableOpReadVariableOp.read_28_disablecopyonread_streaming_dense_bias^Read_28/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:
*
dtype0k
Identity_56IdentityRead_28/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:
a
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0*
_output_shapes
:
�
Read_29/DisableCopyOnReadDisableCopyOnRead2read_29_disablecopyonread_streaming_dense_1_kernel"/device:CPU:0*
_output_shapes
 �
Read_29/ReadVariableOpReadVariableOp2read_29_disablecopyonread_streaming_dense_1_kernel^Read_29/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:
*
dtype0o
Identity_58IdentityRead_29/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:
e
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*
_output_shapes

:
�
Read_30/DisableCopyOnReadDisableCopyOnRead0read_30_disablecopyonread_streaming_dense_1_bias"/device:CPU:0*
_output_shapes
 �
Read_30/ReadVariableOpReadVariableOp0read_30_disablecopyonread_streaming_dense_1_bias^Read_30/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_60IdentityRead_30/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_61IdentityIdentity_60:output:0"/device:CPU:0*
T0*
_output_shapes
:|
Read_31/DisableCopyOnReadDisableCopyOnRead'read_31_disablecopyonread_stream_kernel"/device:CPU:0*
_output_shapes
 �
Read_31/ReadVariableOpReadVariableOp'read_31_disablecopyonread_stream_kernel^Read_31/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0w
Identity_62IdentityRead_31/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:m
Identity_63IdentityIdentity_62:output:0"/device:CPU:0*
T0*&
_output_shapes
:z
Read_32/DisableCopyOnReadDisableCopyOnRead%read_32_disablecopyonread_stream_bias"/device:CPU:0*
_output_shapes
 �
Read_32/ReadVariableOpReadVariableOp%read_32_disablecopyonread_stream_bias^Read_32/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_64IdentityRead_32/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_65IdentityIdentity_64:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_33/DisableCopyOnReadDisableCopyOnRead3read_33_disablecopyonread_stream_1_depthwise_kernel"/device:CPU:0*
_output_shapes
 �
Read_33/ReadVariableOpReadVariableOp3read_33_disablecopyonread_stream_1_depthwise_kernel^Read_33/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0w
Identity_66IdentityRead_33/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:m
Identity_67IdentityIdentity_66:output:0"/device:CPU:0*
T0*&
_output_shapes
:|
Read_34/DisableCopyOnReadDisableCopyOnRead'read_34_disablecopyonread_stream_1_bias"/device:CPU:0*
_output_shapes
 �
Read_34/ReadVariableOpReadVariableOp'read_34_disablecopyonread_stream_1_bias^Read_34/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_68IdentityRead_34/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_69IdentityIdentity_68:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_35/DisableCopyOnReadDisableCopyOnRead3read_35_disablecopyonread_stream_2_depthwise_kernel"/device:CPU:0*
_output_shapes
 �
Read_35/ReadVariableOpReadVariableOp3read_35_disablecopyonread_stream_2_depthwise_kernel^Read_35/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0w
Identity_70IdentityRead_35/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:m
Identity_71IdentityIdentity_70:output:0"/device:CPU:0*
T0*&
_output_shapes
:|
Read_36/DisableCopyOnReadDisableCopyOnRead'read_36_disablecopyonread_stream_2_bias"/device:CPU:0*
_output_shapes
 �
Read_36/ReadVariableOpReadVariableOp'read_36_disablecopyonread_stream_2_bias^Read_36/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_72IdentityRead_36/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_73IdentityIdentity_72:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_37/DisableCopyOnReadDisableCopyOnRead3read_37_disablecopyonread_stream_4_depthwise_kernel"/device:CPU:0*
_output_shapes
 �
Read_37/ReadVariableOpReadVariableOp3read_37_disablecopyonread_stream_4_depthwise_kernel^Read_37/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0w
Identity_74IdentityRead_37/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:m
Identity_75IdentityIdentity_74:output:0"/device:CPU:0*
T0*&
_output_shapes
:|
Read_38/DisableCopyOnReadDisableCopyOnRead'read_38_disablecopyonread_stream_4_bias"/device:CPU:0*
_output_shapes
 �
Read_38/ReadVariableOpReadVariableOp'read_38_disablecopyonread_stream_4_bias^Read_38/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_76IdentityRead_38/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_77IdentityIdentity_76:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_39/DisableCopyOnReadDisableCopyOnRead3read_39_disablecopyonread_stream_5_depthwise_kernel"/device:CPU:0*
_output_shapes
 �
Read_39/ReadVariableOpReadVariableOp3read_39_disablecopyonread_stream_5_depthwise_kernel^Read_39/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0w
Identity_78IdentityRead_39/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:m
Identity_79IdentityIdentity_78:output:0"/device:CPU:0*
T0*&
_output_shapes
:|
Read_40/DisableCopyOnReadDisableCopyOnRead'read_40_disablecopyonread_stream_5_bias"/device:CPU:0*
_output_shapes
 �
Read_40/ReadVariableOpReadVariableOp'read_40_disablecopyonread_stream_5_bias^Read_40/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_80IdentityRead_40/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_81IdentityIdentity_80:output:0"/device:CPU:0*
T0*
_output_shapes
:�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:**
dtype0*�
value�B�*B6layer_with_weights-0/states/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/states/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/states/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/states/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/states/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/states/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/states/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:**
dtype0*g
value^B\*B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �	
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0Identity_61:output:0Identity_63:output:0Identity_65:output:0Identity_67:output:0Identity_69:output:0Identity_71:output:0Identity_73:output:0Identity_75:output:0Identity_77:output:0Identity_79:output:0Identity_81:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *8
dtypes.
,2*�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_82Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_83IdentityIdentity_82:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_30/DisableCopyOnRead^Read_30/ReadVariableOp^Read_31/DisableCopyOnRead^Read_31/ReadVariableOp^Read_32/DisableCopyOnRead^Read_32/ReadVariableOp^Read_33/DisableCopyOnRead^Read_33/ReadVariableOp^Read_34/DisableCopyOnRead^Read_34/ReadVariableOp^Read_35/DisableCopyOnRead^Read_35/ReadVariableOp^Read_36/DisableCopyOnRead^Read_36/ReadVariableOp^Read_37/DisableCopyOnRead^Read_37/ReadVariableOp^Read_38/DisableCopyOnRead^Read_38/ReadVariableOp^Read_39/DisableCopyOnRead^Read_39/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_40/DisableCopyOnRead^Read_40/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "#
identity_83Identity_83:output:0*i
_input_shapesX
V: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp26
Read_25/DisableCopyOnReadRead_25/DisableCopyOnRead20
Read_25/ReadVariableOpRead_25/ReadVariableOp26
Read_26/DisableCopyOnReadRead_26/DisableCopyOnRead20
Read_26/ReadVariableOpRead_26/ReadVariableOp26
Read_27/DisableCopyOnReadRead_27/DisableCopyOnRead20
Read_27/ReadVariableOpRead_27/ReadVariableOp26
Read_28/DisableCopyOnReadRead_28/DisableCopyOnRead20
Read_28/ReadVariableOpRead_28/ReadVariableOp26
Read_29/DisableCopyOnReadRead_29/DisableCopyOnRead20
Read_29/ReadVariableOpRead_29/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp26
Read_30/DisableCopyOnReadRead_30/DisableCopyOnRead20
Read_30/ReadVariableOpRead_30/ReadVariableOp26
Read_31/DisableCopyOnReadRead_31/DisableCopyOnRead20
Read_31/ReadVariableOpRead_31/ReadVariableOp26
Read_32/DisableCopyOnReadRead_32/DisableCopyOnRead20
Read_32/ReadVariableOpRead_32/ReadVariableOp26
Read_33/DisableCopyOnReadRead_33/DisableCopyOnRead20
Read_33/ReadVariableOpRead_33/ReadVariableOp26
Read_34/DisableCopyOnReadRead_34/DisableCopyOnRead20
Read_34/ReadVariableOpRead_34/ReadVariableOp26
Read_35/DisableCopyOnReadRead_35/DisableCopyOnRead20
Read_35/ReadVariableOpRead_35/ReadVariableOp26
Read_36/DisableCopyOnReadRead_36/DisableCopyOnRead20
Read_36/ReadVariableOpRead_36/ReadVariableOp26
Read_37/DisableCopyOnReadRead_37/DisableCopyOnRead20
Read_37/ReadVariableOpRead_37/ReadVariableOp26
Read_38/DisableCopyOnReadRead_38/DisableCopyOnRead20
Read_38/ReadVariableOpRead_38/ReadVariableOp26
Read_39/DisableCopyOnReadRead_39/DisableCopyOnRead20
Read_39/ReadVariableOpRead_39/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp26
Read_40/DisableCopyOnReadRead_40/DisableCopyOnRead20
Read_40/ReadVariableOpRead_40/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:*

_output_shapes
: :C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
F
*__inference_activation_layer_call_fn_12419

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:$* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_10740_
IdentityIdentityPartitionedCall:output:0*
T0*&
_output_shapes
:$"
identityIdentity:output:0*%
_input_shapes
:$:N J
&
_output_shapes
:$
 
_user_specified_nameinputs
�
H
,__inference_activation_3_layer_call_fn_12714

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_activation_3_layer_call_and_return_conditional_losses_10867_
IdentityIdentityPartitionedCall:output:0*
T0*&
_output_shapes
:"
identityIdentity:output:0*%
_input_shapes
::N J
&
_output_shapes
:
 
_user_specified_nameinputs
�
�
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_10494

inputsB
4readvariableop_streaming_batch_normalization_2_gamma:C
5readvariableop_1_streaming_batch_normalization_2_beta:Y
Kfusedbatchnormv3_readvariableop_streaming_batch_normalization_2_moving_mean:_
Qfusedbatchnormv3_readvariableop_1_streaming_batch_normalization_2_moving_variance:
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1
ReadVariableOpReadVariableOp4readvariableop_streaming_batch_normalization_2_gamma*
_output_shapes
:*
dtype0�
ReadVariableOp_1ReadVariableOp5readvariableop_1_streaming_batch_normalization_2_beta*
_output_shapes
:*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOpKfusedbatchnormv3_readvariableop_streaming_batch_normalization_2_moving_mean*
_output_shapes
:*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpQfusedbatchnormv3_readvariableop_1_streaming_batch_normalization_2_moving_variance*
_output_shapes
:*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%��'7*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
N__inference_batch_normalization_layer_call_and_return_conditional_losses_12414

inputs@
2readvariableop_streaming_batch_normalization_gamma:A
3readvariableop_1_streaming_batch_normalization_beta:W
Ifusedbatchnormv3_readvariableop_streaming_batch_normalization_moving_mean:]
Ofusedbatchnormv3_readvariableop_1_streaming_batch_normalization_moving_variance:
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1}
ReadVariableOpReadVariableOp2readvariableop_streaming_batch_normalization_gamma*
_output_shapes
:*
dtype0�
ReadVariableOp_1ReadVariableOp3readvariableop_1_streaming_batch_normalization_beta*
_output_shapes
:*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOpIfusedbatchnormv3_readvariableop_streaming_batch_normalization_moving_mean*
_output_shapes
:*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOfusedbatchnormv3_readvariableop_1_streaming_batch_normalization_moving_variance*
_output_shapes
:*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%��'7*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�	
a
B__inference_dropout_layer_call_and_return_conditional_losses_12833

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @c
dropout/MulMulinputsdropout/Const:output:0*
T0*&
_output_shapes
:f
dropout/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            �
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*&
_output_shapes
:*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*&
_output_shapes
:T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*&
_output_shapes
:`
IdentityIdentitydropout/SelectV2:output:0*
T0*&
_output_shapes
:"
identityIdentity:output:0*%
_input_shapes
::N J
&
_output_shapes
:
 
_user_specified_nameinputs
�
H
,__inference_activation_1_layer_call_fn_12511

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_10777_
IdentityIdentityPartitionedCall:output:0*
T0*&
_output_shapes
:"
identityIdentity:output:0*%
_input_shapes
::N J
&
_output_shapes
:
 
_user_specified_nameinputs
�
�
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_12506

inputsB
4readvariableop_streaming_batch_normalization_1_gamma:C
5readvariableop_1_streaming_batch_normalization_1_beta:Y
Kfusedbatchnormv3_readvariableop_streaming_batch_normalization_1_moving_mean:_
Qfusedbatchnormv3_readvariableop_1_streaming_batch_normalization_1_moving_variance:
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1
ReadVariableOpReadVariableOp4readvariableop_streaming_batch_normalization_1_gamma*
_output_shapes
:*
dtype0�
ReadVariableOp_1ReadVariableOp5readvariableop_1_streaming_batch_normalization_1_beta*
_output_shapes
:*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOpKfusedbatchnormv3_readvariableop_streaming_batch_normalization_1_moving_mean*
_output_shapes
:*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpQfusedbatchnormv3_readvariableop_1_streaming_batch_normalization_1_moving_variance*
_output_shapes
:*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%��'7*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_12709

inputsB
4readvariableop_streaming_batch_normalization_3_gamma:C
5readvariableop_1_streaming_batch_normalization_3_beta:Y
Kfusedbatchnormv3_readvariableop_streaming_batch_normalization_3_moving_mean:_
Qfusedbatchnormv3_readvariableop_1_streaming_batch_normalization_3_moving_variance:
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1
ReadVariableOpReadVariableOp4readvariableop_streaming_batch_normalization_3_gamma*
_output_shapes
:*
dtype0�
ReadVariableOp_1ReadVariableOp5readvariableop_1_streaming_batch_normalization_3_beta*
_output_shapes
:*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOpKfusedbatchnormv3_readvariableop_streaming_batch_normalization_3_moving_mean*
_output_shapes
:*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpQfusedbatchnormv3_readvariableop_1_streaming_batch_normalization_3_moving_variance*
_output_shapes
:*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%��'7*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
d
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_12334

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
h
L__inference_average_pooling2d_layer_call_and_return_conditional_losses_12904

inputs
identity�
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
c
G__inference_activation_4_layer_call_and_return_conditional_losses_10904

inputs
identityE
ReluReluinputs*
T0*&
_output_shapes
:Y
IdentityIdentityRelu:activations:0*
T0*&
_output_shapes
:"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
::N J
&
_output_shapes
:
 
_user_specified_nameinputs
�!
�
%__inference_model_layer_call_fn_11887

inputs1
streaming_stream_states:'
stream_kernel:
stream_bias:1
#streaming_batch_normalization_gamma:0
"streaming_batch_normalization_beta:7
)streaming_batch_normalization_moving_mean:;
-streaming_batch_normalization_moving_variance:3
streaming_stream_1_states:3
stream_1_depthwise_kernel:
stream_1_bias:3
%streaming_batch_normalization_1_gamma:2
$streaming_batch_normalization_1_beta:9
+streaming_batch_normalization_1_moving_mean:=
/streaming_batch_normalization_1_moving_variance:3
streaming_stream_2_states:3
stream_2_depthwise_kernel:
stream_2_bias:3
%streaming_batch_normalization_2_gamma:2
$streaming_batch_normalization_2_beta:9
+streaming_batch_normalization_2_moving_mean:=
/streaming_batch_normalization_2_moving_variance:3
streaming_stream_3_states:3
streaming_stream_4_states:3
stream_4_depthwise_kernel:
stream_4_bias:3
%streaming_batch_normalization_3_gamma:2
$streaming_batch_normalization_3_beta:9
+streaming_batch_normalization_3_moving_mean:=
/streaming_batch_normalization_3_moving_variance:3
streaming_stream_5_states:3
stream_5_depthwise_kernel:
stream_5_bias:3
%streaming_batch_normalization_4_gamma:2
$streaming_batch_normalization_4_beta:9
+streaming_batch_normalization_4_moving_mean:=
/streaming_batch_normalization_4_moving_variance:3
streaming_stream_6_states:(
streaming_dense_kernel:0
"
streaming_dense_bias:
*
streaming_dense_1_kernel:
$
streaming_dense_1_bias:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstreaming_stream_statesstream_kernelstream_bias#streaming_batch_normalization_gamma"streaming_batch_normalization_beta)streaming_batch_normalization_moving_mean-streaming_batch_normalization_moving_variancestreaming_stream_1_statesstream_1_depthwise_kernelstream_1_bias%streaming_batch_normalization_1_gamma$streaming_batch_normalization_1_beta+streaming_batch_normalization_1_moving_mean/streaming_batch_normalization_1_moving_variancestreaming_stream_2_statesstream_2_depthwise_kernelstream_2_bias%streaming_batch_normalization_2_gamma$streaming_batch_normalization_2_beta+streaming_batch_normalization_2_moving_mean/streaming_batch_normalization_2_moving_variancestreaming_stream_3_statesstreaming_stream_4_statesstream_4_depthwise_kernelstream_4_bias%streaming_batch_normalization_3_gamma$streaming_batch_normalization_3_beta+streaming_batch_normalization_3_moving_mean/streaming_batch_normalization_3_moving_variancestreaming_stream_5_statesstream_5_depthwise_kernelstream_5_bias%streaming_batch_normalization_4_gamma$streaming_batch_normalization_4_beta+streaming_batch_normalization_4_moving_mean/streaming_batch_normalization_4_moving_variancestreaming_stream_6_statesstreaming_dense_kernelstreaming_dense_biasstreaming_dense_1_kernelstreaming_dense_1_bias*5
Tin.
,2**
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*:
_read_only_resource_inputs
	
 !"&'()*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_11108f
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*x
_input_shapesg
e:�: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:�
 
_user_specified_nameinputs
�

a
B__inference_dropout_layer_call_and_return_conditional_losses_10918

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @c
dropout/MulMulinputsdropout/Const:output:0*
T0*&
_output_shapes
:f
dropout/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            �
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*&
_output_shapes
:*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*&
_output_shapes
:T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*&
_output_shapes
:`
IdentityIdentitydropout/SelectV2:output:0*
T0*&
_output_shapes
:"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
::N J
&
_output_shapes
:
 
_user_specified_nameinputs
�
�
C__inference_stream_1_layer_call_and_return_conditional_losses_12452

inputsI
/concat_readvariableop_streaming_stream_1_states:]
Cdepthwise_conv2d_depthwise_readvariableop_stream_1_depthwise_kernel:C
5depthwise_conv2d_biasadd_readvariableop_stream_1_bias:
identity��AssignVariableOp�concat/ReadVariableOp�'depthwise_conv2d/BiasAdd/ReadVariableOp�)depthwise_conv2d/depthwise/ReadVariableOp�
concat/ReadVariableOpReadVariableOp/concat_readvariableop_streaming_stream_1_states*&
_output_shapes
:*
dtype0M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatConcatV2concat/ReadVariableOp:value:0inputsconcat/axis:output:0*
N*
T0*&
_output_shapes
:'h
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ����    j
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            j
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         �
strided_sliceStridedSliceconcat:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*&
_output_shapes
:*

begin_mask*
end_mask�
AssignVariableOpAssignVariableOp/concat_readvariableop_streaming_stream_1_statesstrided_slice:output:0^concat/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
)depthwise_conv2d/depthwise/ReadVariableOpReadVariableOpCdepthwise_conv2d_depthwise_readvariableop_stream_1_depthwise_kernel^AssignVariableOp*&
_output_shapes
:*
dtype0�
 depthwise_conv2d/depthwise/ShapeConst^AssignVariableOp*
_output_shapes
:*
dtype0*%
valueB"            �
(depthwise_conv2d/depthwise/dilation_rateConst^AssignVariableOp*
_output_shapes
:*
dtype0*
valueB"      �
depthwise_conv2d/depthwiseDepthwiseConv2dNativeconcat:output:01depthwise_conv2d/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
:*
paddingVALID*
strides
�
'depthwise_conv2d/BiasAdd/ReadVariableOpReadVariableOp5depthwise_conv2d_biasadd_readvariableop_stream_1_bias^AssignVariableOp*
_output_shapes
:*
dtype0�
depthwise_conv2d/BiasAddBiasAdd#depthwise_conv2d/depthwise:output:0/depthwise_conv2d/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:o
IdentityIdentity!depthwise_conv2d/BiasAdd:output:0^NoOp*
T0*&
_output_shapes
:�
NoOpNoOp^AssignVariableOp^concat/ReadVariableOp(^depthwise_conv2d/BiasAdd/ReadVariableOp*^depthwise_conv2d/depthwise/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*+
_input_shapes
:$: : : 2$
AssignVariableOpAssignVariableOp2.
concat/ReadVariableOpconcat/ReadVariableOp2R
'depthwise_conv2d/BiasAdd/ReadVariableOp'depthwise_conv2d/BiasAdd/ReadVariableOp2V
)depthwise_conv2d/depthwise/ReadVariableOp)depthwise_conv2d/depthwise/ReadVariableOp:N J
&
_output_shapes
:$
 
_user_specified_nameinputs
�!
�
#__inference_signature_wrapper_11841

iegm_input1
streaming_stream_states:'
stream_kernel:
stream_bias:1
#streaming_batch_normalization_gamma:0
"streaming_batch_normalization_beta:7
)streaming_batch_normalization_moving_mean:;
-streaming_batch_normalization_moving_variance:3
streaming_stream_1_states:3
stream_1_depthwise_kernel:
stream_1_bias:3
%streaming_batch_normalization_1_gamma:2
$streaming_batch_normalization_1_beta:9
+streaming_batch_normalization_1_moving_mean:=
/streaming_batch_normalization_1_moving_variance:3
streaming_stream_2_states:3
stream_2_depthwise_kernel:
stream_2_bias:3
%streaming_batch_normalization_2_gamma:2
$streaming_batch_normalization_2_beta:9
+streaming_batch_normalization_2_moving_mean:=
/streaming_batch_normalization_2_moving_variance:3
streaming_stream_3_states:3
streaming_stream_4_states:3
stream_4_depthwise_kernel:
stream_4_bias:3
%streaming_batch_normalization_3_gamma:2
$streaming_batch_normalization_3_beta:9
+streaming_batch_normalization_3_moving_mean:=
/streaming_batch_normalization_3_moving_variance:3
streaming_stream_5_states:3
stream_5_depthwise_kernel:
stream_5_bias:3
%streaming_batch_normalization_4_gamma:2
$streaming_batch_normalization_4_beta:9
+streaming_batch_normalization_4_moving_mean:=
/streaming_batch_normalization_4_moving_variance:3
streaming_stream_6_states:(
streaming_dense_kernel:0
"
streaming_dense_bias:
*
streaming_dense_1_kernel:
$
streaming_dense_1_bias:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCall
iegm_inputstreaming_stream_statesstream_kernelstream_bias#streaming_batch_normalization_gamma"streaming_batch_normalization_beta)streaming_batch_normalization_moving_mean-streaming_batch_normalization_moving_variancestreaming_stream_1_statesstream_1_depthwise_kernelstream_1_bias%streaming_batch_normalization_1_gamma$streaming_batch_normalization_1_beta+streaming_batch_normalization_1_moving_mean/streaming_batch_normalization_1_moving_variancestreaming_stream_2_statesstream_2_depthwise_kernelstream_2_bias%streaming_batch_normalization_2_gamma$streaming_batch_normalization_2_beta+streaming_batch_normalization_2_moving_mean/streaming_batch_normalization_2_moving_variancestreaming_stream_3_statesstreaming_stream_4_statesstream_4_depthwise_kernelstream_4_bias%streaming_batch_normalization_3_gamma$streaming_batch_normalization_3_beta+streaming_batch_normalization_3_moving_mean/streaming_batch_normalization_3_moving_variancestreaming_stream_5_statesstream_5_depthwise_kernelstream_5_bias%streaming_batch_normalization_4_gamma$streaming_batch_normalization_4_beta+streaming_batch_normalization_4_moving_mean/streaming_batch_normalization_4_moving_variancestreaming_stream_6_statesstreaming_dense_kernelstreaming_dense_biasstreaming_dense_1_kernelstreaming_dense_1_bias*5
Tin.
,2**
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*D
_read_only_resource_inputs&
$"	
 !"#$&'()*-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference__wrapped_model_10208f
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*x
_input_shapesg
e:�: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
'
_output_shapes
:�
$
_user_specified_name
IEGM_input
�
M
1__inference_average_pooling2d_layer_call_fn_12899

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_average_pooling2d_layer_call_and_return_conditional_losses_10515�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�

�
5__inference_batch_normalization_3_layer_call_fn_12664

inputs3
%streaming_batch_normalization_3_gamma:2
$streaming_batch_normalization_3_beta:9
+streaming_batch_normalization_3_moving_mean:=
/streaming_batch_normalization_3_moving_variance:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs%streaming_batch_normalization_3_gamma$streaming_batch_normalization_3_beta+streaming_batch_normalization_3_moving_mean/streaming_batch_normalization_3_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_10576�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
N__inference_batch_normalization_layer_call_and_return_conditional_losses_10310

inputs@
2readvariableop_streaming_batch_normalization_gamma:A
3readvariableop_1_streaming_batch_normalization_beta:W
Ifusedbatchnormv3_readvariableop_streaming_batch_normalization_moving_mean:]
Ofusedbatchnormv3_readvariableop_1_streaming_batch_normalization_moving_variance:
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1}
ReadVariableOpReadVariableOp2readvariableop_streaming_batch_normalization_gamma*
_output_shapes
:*
dtype0�
ReadVariableOp_1ReadVariableOp3readvariableop_1_streaming_batch_normalization_beta*
_output_shapes
:*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOpIfusedbatchnormv3_readvariableop_streaming_batch_normalization_moving_mean*
_output_shapes
:*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOfusedbatchnormv3_readvariableop_1_streaming_batch_normalization_moving_variance*
_output_shapes
:*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%��'7*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_10576

inputsB
4readvariableop_streaming_batch_normalization_3_gamma:C
5readvariableop_1_streaming_batch_normalization_3_beta:Y
Kfusedbatchnormv3_readvariableop_streaming_batch_normalization_3_moving_mean:_
Qfusedbatchnormv3_readvariableop_1_streaming_batch_normalization_3_moving_variance:
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1
ReadVariableOpReadVariableOp4readvariableop_streaming_batch_normalization_3_gamma*
_output_shapes
:*
dtype0�
ReadVariableOp_1ReadVariableOp5readvariableop_1_streaming_batch_normalization_3_beta*
_output_shapes
:*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOpKfusedbatchnormv3_readvariableop_streaming_batch_normalization_3_moving_mean*
_output_shapes
:*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpQfusedbatchnormv3_readvariableop_1_streaming_batch_normalization_3_moving_variance*
_output_shapes
:*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%��'7*
exponential_avg_factor%fff?�
AssignNewValueAssignVariableOpKfusedbatchnormv3_readvariableop_streaming_batch_normalization_3_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOpQfusedbatchnormv3_readvariableop_1_streaming_batch_normalization_3_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
C__inference_stream_5_layer_call_and_return_conditional_losses_10889

inputsI
/concat_readvariableop_streaming_stream_5_states:_
Edepthwise_conv2d_3_depthwise_readvariableop_stream_5_depthwise_kernel:E
7depthwise_conv2d_3_biasadd_readvariableop_stream_5_bias:
identity��AssignVariableOp�concat/ReadVariableOp�)depthwise_conv2d_3/BiasAdd/ReadVariableOp�+depthwise_conv2d_3/depthwise/ReadVariableOp�
concat/ReadVariableOpReadVariableOp/concat_readvariableop_streaming_stream_5_states*&
_output_shapes
:*
dtype0M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatConcatV2concat/ReadVariableOp:value:0inputsconcat/axis:output:0*
N*
T0*&
_output_shapes
:h
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ����    j
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            j
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         �
strided_sliceStridedSliceconcat:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*&
_output_shapes
:*

begin_mask*
end_mask�
AssignVariableOpAssignVariableOp/concat_readvariableop_streaming_stream_5_statesstrided_slice:output:0^concat/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
+depthwise_conv2d_3/depthwise/ReadVariableOpReadVariableOpEdepthwise_conv2d_3_depthwise_readvariableop_stream_5_depthwise_kernel^AssignVariableOp*&
_output_shapes
:*
dtype0�
"depthwise_conv2d_3/depthwise/ShapeConst^AssignVariableOp*
_output_shapes
:*
dtype0*%
valueB"            �
*depthwise_conv2d_3/depthwise/dilation_rateConst^AssignVariableOp*
_output_shapes
:*
dtype0*
valueB"      �
depthwise_conv2d_3/depthwiseDepthwiseConv2dNativeconcat:output:03depthwise_conv2d_3/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
:*
paddingVALID*
strides
�
)depthwise_conv2d_3/BiasAdd/ReadVariableOpReadVariableOp7depthwise_conv2d_3_biasadd_readvariableop_stream_5_bias^AssignVariableOp*
_output_shapes
:*
dtype0�
depthwise_conv2d_3/BiasAddBiasAdd%depthwise_conv2d_3/depthwise:output:01depthwise_conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:q
IdentityIdentity#depthwise_conv2d_3/BiasAdd:output:0^NoOp*
T0*&
_output_shapes
:�
NoOpNoOp^AssignVariableOp^concat/ReadVariableOp*^depthwise_conv2d_3/BiasAdd/ReadVariableOp,^depthwise_conv2d_3/depthwise/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:: : : 2$
AssignVariableOpAssignVariableOp2.
concat/ReadVariableOpconcat/ReadVariableOp2V
)depthwise_conv2d_3/BiasAdd/ReadVariableOp)depthwise_conv2d_3/BiasAdd/ReadVariableOp2Z
+depthwise_conv2d_3/depthwise/ReadVariableOp+depthwise_conv2d_3/depthwise/ReadVariableOp:N J
&
_output_shapes
:
 
_user_specified_nameinputs
�
�
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_12783

inputsB
4readvariableop_streaming_batch_normalization_4_gamma:C
5readvariableop_1_streaming_batch_normalization_4_beta:Y
Kfusedbatchnormv3_readvariableop_streaming_batch_normalization_4_moving_mean:_
Qfusedbatchnormv3_readvariableop_1_streaming_batch_normalization_4_moving_variance:
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1
ReadVariableOpReadVariableOp4readvariableop_streaming_batch_normalization_4_gamma*
_output_shapes
:*
dtype0�
ReadVariableOp_1ReadVariableOp5readvariableop_1_streaming_batch_normalization_4_beta*
_output_shapes
:*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOpKfusedbatchnormv3_readvariableop_streaming_batch_normalization_4_moving_mean*
_output_shapes
:*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpQfusedbatchnormv3_readvariableop_1_streaming_batch_normalization_4_moving_variance*
_output_shapes
:*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%��'7*
exponential_avg_factor%fff?�
AssignNewValueAssignVariableOpKfusedbatchnormv3_readvariableop_streaming_batch_normalization_4_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOpQfusedbatchnormv3_readvariableop_1_streaming_batch_normalization_4_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
c
G__inference_activation_1_layer_call_and_return_conditional_losses_12516

inputs
identityE
ReluReluinputs*
T0*&
_output_shapes
:Y
IdentityIdentityRelu:activations:0*
T0*&
_output_shapes
:"
identityIdentity:output:0*%
_input_shapes
::N J
&
_output_shapes
:
 
_user_specified_nameinputs
�
�
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_12801

inputsB
4readvariableop_streaming_batch_normalization_4_gamma:C
5readvariableop_1_streaming_batch_normalization_4_beta:Y
Kfusedbatchnormv3_readvariableop_streaming_batch_normalization_4_moving_mean:_
Qfusedbatchnormv3_readvariableop_1_streaming_batch_normalization_4_moving_variance:
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1
ReadVariableOpReadVariableOp4readvariableop_streaming_batch_normalization_4_gamma*
_output_shapes
:*
dtype0�
ReadVariableOp_1ReadVariableOp5readvariableop_1_streaming_batch_normalization_4_beta*
_output_shapes
:*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOpKfusedbatchnormv3_readvariableop_streaming_batch_normalization_4_moving_mean*
_output_shapes
:*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpQfusedbatchnormv3_readvariableop_1_streaming_batch_normalization_4_moving_variance*
_output_shapes
:*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%��'7*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
A__inference_stream_layer_call_and_return_conditional_losses_10725

inputsG
-concat_readvariableop_streaming_stream_states:D
*conv2d_conv2d_readvariableop_stream_kernel:7
)conv2d_biasadd_readvariableop_stream_bias:
identity��AssignVariableOp�concat/ReadVariableOp�conv2d/BiasAdd/ReadVariableOp�conv2d/Conv2D/ReadVariableOp�
concat/ReadVariableOpReadVariableOp-concat_readvariableop_streaming_stream_states*&
_output_shapes
:*
dtype0M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatConcatV2concat/ReadVariableOp:value:0inputsconcat/axis:output:0*
N*
T0*&
_output_shapes
:Lh
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ����    j
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            j
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         �
strided_sliceStridedSliceconcat:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*&
_output_shapes
:*

begin_mask*
end_mask�
AssignVariableOpAssignVariableOp-concat_readvariableop_streaming_stream_statesstrided_slice:output:0^concat/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
conv2d/Conv2D/ReadVariableOpReadVariableOp*conv2d_conv2d_readvariableop_stream_kernel^AssignVariableOp*&
_output_shapes
:*
dtype0�
conv2d/Conv2DConv2Dconcat:output:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:$*
paddingVALID*
strides
�
conv2d/BiasAdd/ReadVariableOpReadVariableOp)conv2d_biasadd_readvariableop_stream_bias^AssignVariableOp*
_output_shapes
:*
dtype0�
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:$e
IdentityIdentityconv2d/BiasAdd:output:0^NoOp*
T0*&
_output_shapes
:$�
NoOpNoOp^AssignVariableOp^concat/ReadVariableOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:H: : : 2$
AssignVariableOpAssignVariableOp2.
concat/ReadVariableOpconcat/ReadVariableOp2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp:N J
&
_output_shapes
:H
 
_user_specified_nameinputs
�
�
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_10402

inputsB
4readvariableop_streaming_batch_normalization_1_gamma:C
5readvariableop_1_streaming_batch_normalization_1_beta:Y
Kfusedbatchnormv3_readvariableop_streaming_batch_normalization_1_moving_mean:_
Qfusedbatchnormv3_readvariableop_1_streaming_batch_normalization_1_moving_variance:
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1
ReadVariableOpReadVariableOp4readvariableop_streaming_batch_normalization_1_gamma*
_output_shapes
:*
dtype0�
ReadVariableOp_1ReadVariableOp5readvariableop_1_streaming_batch_normalization_1_beta*
_output_shapes
:*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOpKfusedbatchnormv3_readvariableop_streaming_batch_normalization_1_moving_mean*
_output_shapes
:*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpQfusedbatchnormv3_readvariableop_1_streaming_batch_normalization_1_moving_variance*
_output_shapes
:*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%��'7*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�

�
5__inference_batch_normalization_2_layer_call_fn_12553

inputs3
%streaming_batch_normalization_2_gamma:2
$streaming_batch_normalization_2_beta:9
+streaming_batch_normalization_2_moving_mean:=
/streaming_batch_normalization_2_moving_variance:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs%streaming_batch_normalization_2_gamma$streaming_batch_normalization_2_beta+streaming_batch_normalization_2_moving_mean/streaming_batch_normalization_2_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_10467�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�

�
B__inference_dense_1_layer_call_and_return_conditional_losses_10963

inputs@
.matmul_readvariableop_streaming_dense_1_kernel:
;
-biasadd_readvariableop_streaming_dense_1_bias:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOp.matmul_readvariableop_streaming_dense_1_kernel*
_output_shapes

:
*
dtype0`
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:�
BiasAdd/ReadVariableOpReadVariableOp-biasadd_readvariableop_streaming_dense_1_bias*
_output_shapes
:*
dtype0m
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:M
SoftmaxSoftmaxBiasAdd:output:0*
T0*
_output_shapes

:W
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*
_output_shapes

:w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
:
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:F B

_output_shapes

:

 
_user_specified_nameinputs
�
H
,__inference_activation_2_layer_call_fn_12603

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_activation_2_layer_call_and_return_conditional_losses_10814_
IdentityIdentityPartitionedCall:output:0*
T0*&
_output_shapes
:"
identityIdentity:output:0*%
_input_shapes
::N J
&
_output_shapes
:
 
_user_specified_nameinputs
�
�
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_12580

inputsB
4readvariableop_streaming_batch_normalization_2_gamma:C
5readvariableop_1_streaming_batch_normalization_2_beta:Y
Kfusedbatchnormv3_readvariableop_streaming_batch_normalization_2_moving_mean:_
Qfusedbatchnormv3_readvariableop_1_streaming_batch_normalization_2_moving_variance:
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1
ReadVariableOpReadVariableOp4readvariableop_streaming_batch_normalization_2_gamma*
_output_shapes
:*
dtype0�
ReadVariableOp_1ReadVariableOp5readvariableop_1_streaming_batch_normalization_2_beta*
_output_shapes
:*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOpKfusedbatchnormv3_readvariableop_streaming_batch_normalization_2_moving_mean*
_output_shapes
:*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpQfusedbatchnormv3_readvariableop_1_streaming_batch_normalization_2_moving_variance*
_output_shapes
:*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%��'7*
exponential_avg_factor%fff?�
AssignNewValueAssignVariableOpKfusedbatchnormv3_readvariableop_streaming_batch_normalization_2_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOpQfusedbatchnormv3_readvariableop_1_streaming_batch_normalization_2_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_12488

inputsB
4readvariableop_streaming_batch_normalization_1_gamma:C
5readvariableop_1_streaming_batch_normalization_1_beta:Y
Kfusedbatchnormv3_readvariableop_streaming_batch_normalization_1_moving_mean:_
Qfusedbatchnormv3_readvariableop_1_streaming_batch_normalization_1_moving_variance:
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1
ReadVariableOpReadVariableOp4readvariableop_streaming_batch_normalization_1_gamma*
_output_shapes
:*
dtype0�
ReadVariableOp_1ReadVariableOp5readvariableop_1_streaming_batch_normalization_1_beta*
_output_shapes
:*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOpKfusedbatchnormv3_readvariableop_streaming_batch_normalization_1_moving_mean*
_output_shapes
:*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpQfusedbatchnormv3_readvariableop_1_streaming_batch_normalization_1_moving_variance*
_output_shapes
:*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%��'7*
exponential_avg_factor%fff?�
AssignNewValueAssignVariableOpKfusedbatchnormv3_readvariableop_streaming_batch_normalization_1_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOpQfusedbatchnormv3_readvariableop_1_streaming_batch_normalization_1_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
`
B__inference_dropout_layer_call_and_return_conditional_losses_12838

inputs

identity_1M
IdentityIdentityinputs*
T0*&
_output_shapes
:Z

Identity_1IdentityIdentity:output:0*
T0*&
_output_shapes
:"!

identity_1Identity_1:output:0*%
_input_shapes
::N J
&
_output_shapes
:
 
_user_specified_nameinputs
�
c
G__inference_activation_4_layer_call_and_return_conditional_losses_12811

inputs
identityE
ReluReluinputs*
T0*&
_output_shapes
:Y
IdentityIdentityRelu:activations:0*
T0*&
_output_shapes
:"
identityIdentity:output:0*%
_input_shapes
::N J
&
_output_shapes
:
 
_user_specified_nameinputs
�
�
'__inference_dense_1_layer_call_fn_12883

inputs*
streaming_dense_1_kernel:
$
streaming_dense_1_bias:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstreaming_dense_1_kernelstreaming_dense_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_10963f
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*!
_input_shapes
:
: : 22
StatefulPartitionedCallStatefulPartitionedCall:F B

_output_shapes

:

 
_user_specified_nameinputs
�
�
C__inference_stream_2_layer_call_and_return_conditional_losses_10799

inputsI
/concat_readvariableop_streaming_stream_2_states:_
Edepthwise_conv2d_1_depthwise_readvariableop_stream_2_depthwise_kernel:E
7depthwise_conv2d_1_biasadd_readvariableop_stream_2_bias:
identity��AssignVariableOp�concat/ReadVariableOp�)depthwise_conv2d_1/BiasAdd/ReadVariableOp�+depthwise_conv2d_1/depthwise/ReadVariableOp�
concat/ReadVariableOpReadVariableOp/concat_readvariableop_streaming_stream_2_states*&
_output_shapes
:*
dtype0M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatConcatV2concat/ReadVariableOp:value:0inputsconcat/axis:output:0*
N*
T0*&
_output_shapes
:h
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ����    j
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            j
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         �
strided_sliceStridedSliceconcat:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*&
_output_shapes
:*

begin_mask*
end_mask�
AssignVariableOpAssignVariableOp/concat_readvariableop_streaming_stream_2_statesstrided_slice:output:0^concat/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
+depthwise_conv2d_1/depthwise/ReadVariableOpReadVariableOpEdepthwise_conv2d_1_depthwise_readvariableop_stream_2_depthwise_kernel^AssignVariableOp*&
_output_shapes
:*
dtype0�
"depthwise_conv2d_1/depthwise/ShapeConst^AssignVariableOp*
_output_shapes
:*
dtype0*%
valueB"            �
*depthwise_conv2d_1/depthwise/dilation_rateConst^AssignVariableOp*
_output_shapes
:*
dtype0*
valueB"      �
depthwise_conv2d_1/depthwiseDepthwiseConv2dNativeconcat:output:03depthwise_conv2d_1/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
:*
paddingVALID*
strides
�
)depthwise_conv2d_1/BiasAdd/ReadVariableOpReadVariableOp7depthwise_conv2d_1_biasadd_readvariableop_stream_2_bias^AssignVariableOp*
_output_shapes
:*
dtype0�
depthwise_conv2d_1/BiasAddBiasAdd%depthwise_conv2d_1/depthwise:output:01depthwise_conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:q
IdentityIdentity#depthwise_conv2d_1/BiasAdd:output:0^NoOp*
T0*&
_output_shapes
:�
NoOpNoOp^AssignVariableOp^concat/ReadVariableOp*^depthwise_conv2d_1/BiasAdd/ReadVariableOp,^depthwise_conv2d_1/depthwise/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:: : : 2$
AssignVariableOpAssignVariableOp2.
concat/ReadVariableOpconcat/ReadVariableOp2V
)depthwise_conv2d_1/BiasAdd/ReadVariableOp)depthwise_conv2d_1/BiasAdd/ReadVariableOp2Z
+depthwise_conv2d_1/depthwise/ReadVariableOp+depthwise_conv2d_1/depthwise/ReadVariableOp:N J
&
_output_shapes
:
 
_user_specified_nameinputs
�
c
G__inference_activation_2_layer_call_and_return_conditional_losses_10814

inputs
identityE
ReluReluinputs*
T0*&
_output_shapes
:Y
IdentityIdentityRelu:activations:0*
T0*&
_output_shapes
:"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
::N J
&
_output_shapes
:
 
_user_specified_nameinputs
�
�
C__inference_stream_5_layer_call_and_return_conditional_losses_12747

inputsI
/concat_readvariableop_streaming_stream_5_states:_
Edepthwise_conv2d_3_depthwise_readvariableop_stream_5_depthwise_kernel:E
7depthwise_conv2d_3_biasadd_readvariableop_stream_5_bias:
identity��AssignVariableOp�concat/ReadVariableOp�)depthwise_conv2d_3/BiasAdd/ReadVariableOp�+depthwise_conv2d_3/depthwise/ReadVariableOp�
concat/ReadVariableOpReadVariableOp/concat_readvariableop_streaming_stream_5_states*&
_output_shapes
:*
dtype0M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatConcatV2concat/ReadVariableOp:value:0inputsconcat/axis:output:0*
N*
T0*&
_output_shapes
:h
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ����    j
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            j
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         �
strided_sliceStridedSliceconcat:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*&
_output_shapes
:*

begin_mask*
end_mask�
AssignVariableOpAssignVariableOp/concat_readvariableop_streaming_stream_5_statesstrided_slice:output:0^concat/ReadVariableOp*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0*
validate_shape(�
+depthwise_conv2d_3/depthwise/ReadVariableOpReadVariableOpEdepthwise_conv2d_3_depthwise_readvariableop_stream_5_depthwise_kernel^AssignVariableOp*&
_output_shapes
:*
dtype0�
"depthwise_conv2d_3/depthwise/ShapeConst^AssignVariableOp*
_output_shapes
:*
dtype0*%
valueB"            �
*depthwise_conv2d_3/depthwise/dilation_rateConst^AssignVariableOp*
_output_shapes
:*
dtype0*
valueB"      �
depthwise_conv2d_3/depthwiseDepthwiseConv2dNativeconcat:output:03depthwise_conv2d_3/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
:*
paddingVALID*
strides
�
)depthwise_conv2d_3/BiasAdd/ReadVariableOpReadVariableOp7depthwise_conv2d_3_biasadd_readvariableop_stream_5_bias^AssignVariableOp*
_output_shapes
:*
dtype0�
depthwise_conv2d_3/BiasAddBiasAdd%depthwise_conv2d_3/depthwise:output:01depthwise_conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:q
IdentityIdentity#depthwise_conv2d_3/BiasAdd:output:0^NoOp*
T0*&
_output_shapes
:�
NoOpNoOp^AssignVariableOp^concat/ReadVariableOp*^depthwise_conv2d_3/BiasAdd/ReadVariableOp,^depthwise_conv2d_3/depthwise/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*+
_input_shapes
:: : : 2$
AssignVariableOpAssignVariableOp2.
concat/ReadVariableOpconcat/ReadVariableOp2V
)depthwise_conv2d_3/BiasAdd/ReadVariableOp)depthwise_conv2d_3/BiasAdd/ReadVariableOp2Z
+depthwise_conv2d_3/depthwise/ReadVariableOp+depthwise_conv2d_3/depthwise/ReadVariableOp:N J
&
_output_shapes
:
 
_user_specified_nameinputs
�
�
N__inference_batch_normalization_layer_call_and_return_conditional_losses_10283

inputs@
2readvariableop_streaming_batch_normalization_gamma:A
3readvariableop_1_streaming_batch_normalization_beta:W
Ifusedbatchnormv3_readvariableop_streaming_batch_normalization_moving_mean:]
Ofusedbatchnormv3_readvariableop_1_streaming_batch_normalization_moving_variance:
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1}
ReadVariableOpReadVariableOp2readvariableop_streaming_batch_normalization_gamma*
_output_shapes
:*
dtype0�
ReadVariableOp_1ReadVariableOp3readvariableop_1_streaming_batch_normalization_beta*
_output_shapes
:*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOpIfusedbatchnormv3_readvariableop_streaming_batch_normalization_moving_mean*
_output_shapes
:*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOfusedbatchnormv3_readvariableop_1_streaming_batch_normalization_moving_variance*
_output_shapes
:*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%��'7*
exponential_avg_factor%fff?�
AssignNewValueAssignVariableOpIfusedbatchnormv3_readvariableop_streaming_batch_normalization_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOpOfusedbatchnormv3_readvariableop_1_streaming_batch_normalization_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
a
E__inference_activation_layer_call_and_return_conditional_losses_12424

inputs
identityE
ReluReluinputs*
T0*&
_output_shapes
:$Y
IdentityIdentityRelu:activations:0*
T0*&
_output_shapes
:$"
identityIdentity:output:0*%
_input_shapes
:$:N J
&
_output_shapes
:$
 
_user_specified_nameinputs
�

�
@__inference_dense_layer_call_and_return_conditional_losses_10948

inputs>
,matmul_readvariableop_streaming_dense_kernel:0
9
+biasadd_readvariableop_streaming_dense_bias:

identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOp,matmul_readvariableop_streaming_dense_kernel*
_output_shapes

:0
*
dtype0`
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:
~
BiasAdd/ReadVariableOpReadVariableOp+biasadd_readvariableop_streaming_dense_bias*
_output_shapes
:
*
dtype0m
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:
G
ReluReluBiasAdd:output:0*
T0*
_output_shapes

:
X
IdentityIdentityRelu:activations:0^NoOp*
T0*
_output_shapes

:
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
:0: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:F B

_output_shapes

:0
 
_user_specified_nameinputs"�
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
A

IEGM_input3
serving_default_IEGM_input:0�2
dense_1'
StatefulPartitionedCall:0tensorflow/serving/predict:��
�
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer_with_weights-5

layer-9
layer-10
layer_with_weights-6
layer-11
layer_with_weights-7
layer-12
layer_with_weights-8
layer-13
layer-14
layer_with_weights-9
layer-15
layer_with_weights-10
layer-16
layer-17
layer-18
layer_with_weights-11
layer-19
layer_with_weights-12
layer-20
layer_with_weights-13
layer-21
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_network
"
_tf_keras_input_layer
�
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses"
_tf_keras_layer
�
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses
+cell
,state_shape

-states"
_tf_keras_layer
�
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses
4axis
	5gamma
6beta
7moving_mean
8moving_variance"
_tf_keras_layer
�
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses"
_tf_keras_layer
�
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
C__call__
*D&call_and_return_all_conditional_losses
Ecell
Fstate_shape

Gstates"
_tf_keras_layer
�
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
L__call__
*M&call_and_return_all_conditional_losses
Naxis
	Ogamma
Pbeta
Qmoving_mean
Rmoving_variance"
_tf_keras_layer
�
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W__call__
*X&call_and_return_all_conditional_losses"
_tf_keras_layer
�
Y	variables
Ztrainable_variables
[regularization_losses
\	keras_api
]__call__
*^&call_and_return_all_conditional_losses
_cell
`state_shape

astates"
_tf_keras_layer
�
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
f__call__
*g&call_and_return_all_conditional_losses
haxis
	igamma
jbeta
kmoving_mean
lmoving_variance"
_tf_keras_layer
�
m	variables
ntrainable_variables
oregularization_losses
p	keras_api
q__call__
*r&call_and_return_all_conditional_losses"
_tf_keras_layer
�
s	variables
ttrainable_variables
uregularization_losses
v	keras_api
w__call__
*x&call_and_return_all_conditional_losses
ycell
zstate_shape

{states"
_tf_keras_layer
�
|	variables
}trainable_variables
~regularization_losses
	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�cell
�state_shape
�states"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�cell
�state_shape
�states"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�cell
�state_shape
�states"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias"
_tf_keras_layer
�
�0
�1
-2
53
64
75
86
�7
�8
G9
O10
P11
Q12
R13
�14
�15
a16
i17
j18
k19
l20
{21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40"
trackable_list_wrapper
�
�0
�1
52
63
�4
�5
O6
P7
�8
�9
i10
j11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_1
�trace_2
�trace_32�
%__inference_model_layer_call_fn_11152
%__inference_model_layer_call_fn_11264
%__inference_model_layer_call_fn_11887
%__inference_model_layer_call_fn_11933�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3
�
�trace_0
�trace_1
�trace_2
�trace_32�
@__inference_model_layer_call_and_return_conditional_losses_10968
@__inference_model_layer_call_and_return_conditional_losses_11039
@__inference_model_layer_call_and_return_conditional_losses_12132
@__inference_model_layer_call_and_return_conditional_losses_12324�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3
�B�
 __inference__wrapped_model_10208
IEGM_input"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
-
�serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
-__inference_max_pooling2d_layer_call_fn_12329�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_12334�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
7
�0
�1
-2"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
&__inference_stream_layer_call_fn_12342�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
A__inference_stream_layer_call_and_return_conditional_losses_12360�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
 "
trackable_list_wrapper
/:-2streaming/stream/states
<
50
61
72
83"
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
3__inference_batch_normalization_layer_call_fn_12369
3__inference_batch_normalization_layer_call_fn_12378�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
N__inference_batch_normalization_layer_call_and_return_conditional_losses_12396
N__inference_batch_normalization_layer_call_and_return_conditional_losses_12414�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
1:/2#streaming/batch_normalization/gamma
0:.2"streaming/batch_normalization/beta
9:7 (2)streaming/batch_normalization/moving_mean
=:; (2-streaming/batch_normalization/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_activation_layer_call_fn_12419�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_activation_layer_call_and_return_conditional_losses_12424�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
7
�0
�1
G2"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
?	variables
@trainable_variables
Aregularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_stream_1_layer_call_fn_12432�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_stream_1_layer_call_and_return_conditional_losses_12452�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�depthwise_kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
 "
trackable_list_wrapper
1:/2streaming/stream_1/states
<
O0
P1
Q2
R3"
trackable_list_wrapper
.
O0
P1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
H	variables
Itrainable_variables
Jregularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
5__inference_batch_normalization_1_layer_call_fn_12461
5__inference_batch_normalization_1_layer_call_fn_12470�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_12488
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_12506�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
3:12%streaming/batch_normalization_1/gamma
2:02$streaming/batch_normalization_1/beta
;:9 (2+streaming/batch_normalization_1/moving_mean
?:= (2/streaming/batch_normalization_1/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
,__inference_activation_1_layer_call_fn_12511�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
G__inference_activation_1_layer_call_and_return_conditional_losses_12516�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
7
�0
�1
a2"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
Y	variables
Ztrainable_variables
[regularization_losses
]__call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_stream_2_layer_call_fn_12524�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_stream_2_layer_call_and_return_conditional_losses_12544�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�depthwise_kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
 "
trackable_list_wrapper
1:/2streaming/stream_2/states
<
i0
j1
k2
l3"
trackable_list_wrapper
.
i0
j1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
b	variables
ctrainable_variables
dregularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
5__inference_batch_normalization_2_layer_call_fn_12553
5__inference_batch_normalization_2_layer_call_fn_12562�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_12580
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_12598�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
3:12%streaming/batch_normalization_2/gamma
2:02$streaming/batch_normalization_2/beta
;:9 (2+streaming/batch_normalization_2/moving_mean
?:= (2/streaming/batch_normalization_2/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
m	variables
ntrainable_variables
oregularization_losses
q__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
,__inference_activation_2_layer_call_fn_12603�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
G__inference_activation_2_layer_call_and_return_conditional_losses_12608�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
'
{0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
s	variables
ttrainable_variables
uregularization_losses
w__call__
*x&call_and_return_all_conditional_losses
&x"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_stream_3_layer_call_fn_12614�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_stream_3_layer_call_and_return_conditional_losses_12627�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
1:/2streaming/stream_3/states
8
�0
�1
�2"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
|	variables
}trainable_variables
~regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_stream_4_layer_call_fn_12635�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_stream_4_layer_call_and_return_conditional_losses_12655�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�depthwise_kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
 "
trackable_list_wrapper
1:/2streaming/stream_4/states
@
�0
�1
�2
�3"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
5__inference_batch_normalization_3_layer_call_fn_12664
5__inference_batch_normalization_3_layer_call_fn_12673�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_12691
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_12709�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
3:12%streaming/batch_normalization_3/gamma
2:02$streaming/batch_normalization_3/beta
;:9 (2+streaming/batch_normalization_3/moving_mean
?:= (2/streaming/batch_normalization_3/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
,__inference_activation_3_layer_call_fn_12714�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
G__inference_activation_3_layer_call_and_return_conditional_losses_12719�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
8
�0
�1
�2"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_stream_5_layer_call_fn_12727�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_stream_5_layer_call_and_return_conditional_losses_12747�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�depthwise_kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
 "
trackable_list_wrapper
1:/2streaming/stream_5/states
@
�0
�1
�2
�3"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
5__inference_batch_normalization_4_layer_call_fn_12756
5__inference_batch_normalization_4_layer_call_fn_12765�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_12783
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_12801�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
3:12%streaming/batch_normalization_4/gamma
2:02$streaming/batch_normalization_4/beta
;:9 (2+streaming/batch_normalization_4/moving_mean
?:= (2/streaming/batch_normalization_4/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
,__inference_activation_4_layer_call_fn_12806�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
G__inference_activation_4_layer_call_and_return_conditional_losses_12811�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
'__inference_dropout_layer_call_fn_12816
'__inference_dropout_layer_call_fn_12821�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
B__inference_dropout_layer_call_and_return_conditional_losses_12833
B__inference_dropout_layer_call_and_return_conditional_losses_12838�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
"
_generic_user_object
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_stream_6_layer_call_fn_12844�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_stream_6_layer_call_and_return_conditional_losses_12858�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
1:/2streaming/stream_6/states
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
%__inference_dense_layer_call_fn_12865�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
@__inference_dense_layer_call_and_return_conditional_losses_12876�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
(:&0
2streaming/dense/kernel
": 
2streaming/dense/bias
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
'__inference_dense_1_layer_call_fn_12883�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
B__inference_dense_1_layer_call_and_return_conditional_losses_12894�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
*:(
2streaming/dense_1/kernel
$:"2streaming/dense_1/bias
':%2stream/kernel
:2stream/bias
3:12stream_1/depthwise_kernel
:2stream_1/bias
3:12stream_2/depthwise_kernel
:2stream_2/bias
3:12stream_4/depthwise_kernel
:2stream_4/bias
3:12stream_5/depthwise_kernel
:2stream_5/bias
�
-0
71
82
G3
Q4
R5
a6
k7
l8
{9
�10
�11
�12
�13
�14
�15
�16"
trackable_list_wrapper
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
%__inference_model_layer_call_fn_11152
IEGM_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
%__inference_model_layer_call_fn_11264
IEGM_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
%__inference_model_layer_call_fn_11887inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
%__inference_model_layer_call_fn_11933inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
@__inference_model_layer_call_and_return_conditional_losses_10968
IEGM_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
@__inference_model_layer_call_and_return_conditional_losses_11039
IEGM_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
@__inference_model_layer_call_and_return_conditional_losses_12132inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
@__inference_model_layer_call_and_return_conditional_losses_12324inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference_signature_wrapper_11841
IEGM_input"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
-__inference_max_pooling2d_layer_call_fn_12329inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_12334inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
'
-0"
trackable_list_wrapper
'
+0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
&__inference_stream_layer_call_fn_12342inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_stream_layer_call_and_return_conditional_losses_12360inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
.
70
81"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
3__inference_batch_normalization_layer_call_fn_12369inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
3__inference_batch_normalization_layer_call_fn_12378inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
N__inference_batch_normalization_layer_call_and_return_conditional_losses_12396inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
N__inference_batch_normalization_layer_call_and_return_conditional_losses_12414inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_activation_layer_call_fn_12419inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_activation_layer_call_and_return_conditional_losses_12424inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
'
G0"
trackable_list_wrapper
'
E0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_stream_1_layer_call_fn_12432inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_stream_1_layer_call_and_return_conditional_losses_12452inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
.
Q0
R1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
5__inference_batch_normalization_1_layer_call_fn_12461inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
5__inference_batch_normalization_1_layer_call_fn_12470inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_12488inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_12506inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_activation_1_layer_call_fn_12511inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_activation_1_layer_call_and_return_conditional_losses_12516inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
'
a0"
trackable_list_wrapper
'
_0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_stream_2_layer_call_fn_12524inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_stream_2_layer_call_and_return_conditional_losses_12544inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
.
k0
l1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
5__inference_batch_normalization_2_layer_call_fn_12553inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
5__inference_batch_normalization_2_layer_call_fn_12562inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_12580inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_12598inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_activation_2_layer_call_fn_12603inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_activation_2_layer_call_and_return_conditional_losses_12608inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
'
{0"
trackable_list_wrapper
'
y0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_stream_3_layer_call_fn_12614inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_stream_3_layer_call_and_return_conditional_losses_12627inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
1__inference_average_pooling2d_layer_call_fn_12899�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
L__inference_average_pooling2d_layer_call_and_return_conditional_losses_12904�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
(
�0"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_stream_4_layer_call_fn_12635inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_stream_4_layer_call_and_return_conditional_losses_12655inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
5__inference_batch_normalization_3_layer_call_fn_12664inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
5__inference_batch_normalization_3_layer_call_fn_12673inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_12691inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_12709inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_activation_3_layer_call_fn_12714inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_activation_3_layer_call_and_return_conditional_losses_12719inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
(
�0"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_stream_5_layer_call_fn_12727inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_stream_5_layer_call_and_return_conditional_losses_12747inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
5__inference_batch_normalization_4_layer_call_fn_12756inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
5__inference_batch_normalization_4_layer_call_fn_12765inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_12783inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_12801inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_activation_4_layer_call_fn_12806inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_activation_4_layer_call_and_return_conditional_losses_12811inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_dropout_layer_call_fn_12816inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
'__inference_dropout_layer_call_fn_12821inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_dropout_layer_call_and_return_conditional_losses_12833inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_dropout_layer_call_and_return_conditional_losses_12838inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
(
�0"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_stream_6_layer_call_fn_12844inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_stream_6_layer_call_and_return_conditional_losses_12858inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
%__inference_dense_layer_call_fn_12865inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
@__inference_dense_layer_call_and_return_conditional_losses_12876inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_dense_1_layer_call_fn_12883inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_dense_1_layer_call_and_return_conditional_losses_12894inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
1__inference_average_pooling2d_layer_call_fn_12899inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
L__inference_average_pooling2d_layer_call_and_return_conditional_losses_12904inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper�
 __inference__wrapped_model_10208�B-��5678G��OPQRa��ijkl{�������������������3�0
)�&
$�!

IEGM_input�
� "(�%
#
dense_1�
dense_1�
G__inference_activation_1_layer_call_and_return_conditional_losses_12516].�+
$�!
�
inputs
� "+�(
!�
tensor_0
� �
,__inference_activation_1_layer_call_fn_12511R.�+
$�!
�
inputs
� " �
unknown�
G__inference_activation_2_layer_call_and_return_conditional_losses_12608].�+
$�!
�
inputs
� "+�(
!�
tensor_0
� �
,__inference_activation_2_layer_call_fn_12603R.�+
$�!
�
inputs
� " �
unknown�
G__inference_activation_3_layer_call_and_return_conditional_losses_12719].�+
$�!
�
inputs
� "+�(
!�
tensor_0
� �
,__inference_activation_3_layer_call_fn_12714R.�+
$�!
�
inputs
� " �
unknown�
G__inference_activation_4_layer_call_and_return_conditional_losses_12811].�+
$�!
�
inputs
� "+�(
!�
tensor_0
� �
,__inference_activation_4_layer_call_fn_12806R.�+
$�!
�
inputs
� " �
unknown�
E__inference_activation_layer_call_and_return_conditional_losses_12424].�+
$�!
�
inputs$
� "+�(
!�
tensor_0$
� �
*__inference_activation_layer_call_fn_12419R.�+
$�!
�
inputs$
� " �
unknown$�
L__inference_average_pooling2d_layer_call_and_return_conditional_losses_12904�R�O
H�E
C�@
inputs4������������������������������������
� "O�L
E�B
tensor_04������������������������������������
� �
1__inference_average_pooling2d_layer_call_fn_12899�R�O
H�E
C�@
inputs4������������������������������������
� "D�A
unknown4�������������������������������������
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_12488�OPQRQ�N
G�D
:�7
inputs+���������������������������
p

 
� "F�C
<�9
tensor_0+���������������������������
� �
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_12506�OPQRQ�N
G�D
:�7
inputs+���������������������������
p 

 
� "F�C
<�9
tensor_0+���������������������������
� �
5__inference_batch_normalization_1_layer_call_fn_12461�OPQRQ�N
G�D
:�7
inputs+���������������������������
p

 
� ";�8
unknown+����������������������������
5__inference_batch_normalization_1_layer_call_fn_12470�OPQRQ�N
G�D
:�7
inputs+���������������������������
p 

 
� ";�8
unknown+����������������������������
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_12580�ijklQ�N
G�D
:�7
inputs+���������������������������
p

 
� "F�C
<�9
tensor_0+���������������������������
� �
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_12598�ijklQ�N
G�D
:�7
inputs+���������������������������
p 

 
� "F�C
<�9
tensor_0+���������������������������
� �
5__inference_batch_normalization_2_layer_call_fn_12553�ijklQ�N
G�D
:�7
inputs+���������������������������
p

 
� ";�8
unknown+����������������������������
5__inference_batch_normalization_2_layer_call_fn_12562�ijklQ�N
G�D
:�7
inputs+���������������������������
p 

 
� ";�8
unknown+����������������������������
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_12691�����Q�N
G�D
:�7
inputs+���������������������������
p

 
� "F�C
<�9
tensor_0+���������������������������
� �
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_12709�����Q�N
G�D
:�7
inputs+���������������������������
p 

 
� "F�C
<�9
tensor_0+���������������������������
� �
5__inference_batch_normalization_3_layer_call_fn_12664�����Q�N
G�D
:�7
inputs+���������������������������
p

 
� ";�8
unknown+����������������������������
5__inference_batch_normalization_3_layer_call_fn_12673�����Q�N
G�D
:�7
inputs+���������������������������
p 

 
� ";�8
unknown+����������������������������
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_12783�����Q�N
G�D
:�7
inputs+���������������������������
p

 
� "F�C
<�9
tensor_0+���������������������������
� �
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_12801�����Q�N
G�D
:�7
inputs+���������������������������
p 

 
� "F�C
<�9
tensor_0+���������������������������
� �
5__inference_batch_normalization_4_layer_call_fn_12756�����Q�N
G�D
:�7
inputs+���������������������������
p

 
� ";�8
unknown+����������������������������
5__inference_batch_normalization_4_layer_call_fn_12765�����Q�N
G�D
:�7
inputs+���������������������������
p 

 
� ";�8
unknown+����������������������������
N__inference_batch_normalization_layer_call_and_return_conditional_losses_12396�5678Q�N
G�D
:�7
inputs+���������������������������
p

 
� "F�C
<�9
tensor_0+���������������������������
� �
N__inference_batch_normalization_layer_call_and_return_conditional_losses_12414�5678Q�N
G�D
:�7
inputs+���������������������������
p 

 
� "F�C
<�9
tensor_0+���������������������������
� �
3__inference_batch_normalization_layer_call_fn_12369�5678Q�N
G�D
:�7
inputs+���������������������������
p

 
� ";�8
unknown+����������������������������
3__inference_batch_normalization_layer_call_fn_12378�5678Q�N
G�D
:�7
inputs+���������������������������
p 

 
� ";�8
unknown+����������������������������
B__inference_dense_1_layer_call_and_return_conditional_losses_12894S��&�#
�
�
inputs

� "#� 
�
tensor_0
� s
'__inference_dense_1_layer_call_fn_12883H��&�#
�
�
inputs

� "�
unknown�
@__inference_dense_layer_call_and_return_conditional_losses_12876S��&�#
�
�
inputs0
� "#� 
�
tensor_0

� q
%__inference_dense_layer_call_fn_12865H��&�#
�
�
inputs0
� "�
unknown
�
B__inference_dropout_layer_call_and_return_conditional_losses_12833a2�/
(�%
�
inputs
p
� "+�(
!�
tensor_0
� �
B__inference_dropout_layer_call_and_return_conditional_losses_12838a2�/
(�%
�
inputs
p 
� "+�(
!�
tensor_0
� �
'__inference_dropout_layer_call_fn_12816V2�/
(�%
�
inputs
p
� " �
unknown�
'__inference_dropout_layer_call_fn_12821V2�/
(�%
�
inputs
p 
� " �
unknown�
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_12334�R�O
H�E
C�@
inputs4������������������������������������
� "O�L
E�B
tensor_04������������������������������������
� �
-__inference_max_pooling2d_layer_call_fn_12329�R�O
H�E
C�@
inputs4������������������������������������
� "D�A
unknown4�������������������������������������
@__inference_model_layer_call_and_return_conditional_losses_10968�B-��5678G��OPQRa��ijkl{�������������������;�8
1�.
$�!

IEGM_input�
p

 
� "#� 
�
tensor_0
� �
@__inference_model_layer_call_and_return_conditional_losses_11039�B-��5678G��OPQRa��ijkl{�������������������;�8
1�.
$�!

IEGM_input�
p 

 
� "#� 
�
tensor_0
� �
@__inference_model_layer_call_and_return_conditional_losses_12132�B-��5678G��OPQRa��ijkl{�������������������7�4
-�*
 �
inputs�
p

 
� "#� 
�
tensor_0
� �
@__inference_model_layer_call_and_return_conditional_losses_12324�B-��5678G��OPQRa��ijkl{�������������������7�4
-�*
 �
inputs�
p 

 
� "#� 
�
tensor_0
� �
%__inference_model_layer_call_fn_11152�B-��5678G��OPQRa��ijkl{�������������������;�8
1�.
$�!

IEGM_input�
p

 
� "�
unknown�
%__inference_model_layer_call_fn_11264�B-��5678G��OPQRa��ijkl{�������������������;�8
1�.
$�!

IEGM_input�
p 

 
� "�
unknown�
%__inference_model_layer_call_fn_11887�B-��5678G��OPQRa��ijkl{�������������������7�4
-�*
 �
inputs�
p

 
� "�
unknown�
%__inference_model_layer_call_fn_11933�B-��5678G��OPQRa��ijkl{�������������������7�4
-�*
 �
inputs�
p 

 
� "�
unknown�
#__inference_signature_wrapper_11841�B-��5678G��OPQRa��ijkl{�������������������A�>
� 
7�4
2

IEGM_input$�!

iegm_input�"(�%
#
dense_1�
dense_1�
C__inference_stream_1_layer_call_and_return_conditional_losses_12452dG��.�+
$�!
�
inputs$
� "+�(
!�
tensor_0
� �
(__inference_stream_1_layer_call_fn_12432YG��.�+
$�!
�
inputs$
� " �
unknown�
C__inference_stream_2_layer_call_and_return_conditional_losses_12544da��.�+
$�!
�
inputs
� "+�(
!�
tensor_0
� �
(__inference_stream_2_layer_call_fn_12524Ya��.�+
$�!
�
inputs
� " �
unknown�
C__inference_stream_3_layer_call_and_return_conditional_losses_12627`{.�+
$�!
�
inputs
� "+�(
!�
tensor_0
� �
(__inference_stream_3_layer_call_fn_12614U{.�+
$�!
�
inputs
� " �
unknown�
C__inference_stream_4_layer_call_and_return_conditional_losses_12655e���.�+
$�!
�
inputs
� "+�(
!�
tensor_0
� �
(__inference_stream_4_layer_call_fn_12635Z���.�+
$�!
�
inputs
� " �
unknown�
C__inference_stream_5_layer_call_and_return_conditional_losses_12747e���.�+
$�!
�
inputs
� "+�(
!�
tensor_0
� �
(__inference_stream_5_layer_call_fn_12727Z���.�+
$�!
�
inputs
� " �
unknown�
C__inference_stream_6_layer_call_and_return_conditional_losses_12858Y�.�+
$�!
�
inputs
� "#� 
�
tensor_00
� z
(__inference_stream_6_layer_call_fn_12844N�.�+
$�!
�
inputs
� "�
unknown0�
A__inference_stream_layer_call_and_return_conditional_losses_12360d-��.�+
$�!
�
inputsH
� "+�(
!�
tensor_0$
� �
&__inference_stream_layer_call_fn_12342Y-��.�+
$�!
�
inputsH
� " �
unknown$