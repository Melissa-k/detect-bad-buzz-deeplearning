??
?$?$
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
K
Bincount
arr
size
weights"T	
bins"T"
Ttype:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
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
?
Cumsum
x"T
axis"Tidx
out"T"
	exclusivebool( "
reversebool( " 
Ttype:
2	"
Tidxtype0:
2	
R
Equal
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(?
=
Greater
x"T
y"T
z
"
Ttype:
2	
?
HashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetype?
.
Identity

input"T
output"T"	
Ttype
l
LookupTableExportV2
table_handle
keys"Tkeys
values"Tvalues"
Tkeystype"
Tvaluestype?
w
LookupTableFindV2
table_handle
keys"Tin
default_value"Tout
values"Tout"
Tintype"
Touttype?
b
LookupTableImportV2
table_handle
keys"Tin
values"Tout"
Tintype"
Touttype?
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
>
Minimum
x"T
y"T
z"T"
Ttype:
2	
?
Mul
x"T
y"T
z"T"
Ttype:
2	?
?
MutableHashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetype?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
?
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
?
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
?
RaggedBincount

splits	
values"Tidx
size"Tidx
weights"T
output"T"
Tidxtype:
2	"
Ttype:
2	"
binary_outputbool( 
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
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
A
SelectV2
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
?
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
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
m
StaticRegexReplace	
input

output"
patternstring"
rewritestring"
replace_globalbool(
?
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
<
StringLower	
input

output"
encodingstring 
e
StringSplitV2	
input
sep
indices	

values	
shape	"
maxsplitint?????????
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.9.22v2.9.1-132-g18960c44ad38??
?
Adam/dense_27/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_27/bias/v
y
(Adam/dense_27/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_27/bias/v*
_output_shapes
:*
dtype0
?
Adam/dense_27/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*'
shared_nameAdam/dense_27/kernel/v
?
*Adam/dense_27/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_27/kernel/v*
_output_shapes

:
*
dtype0
?
Adam/dense_26/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*%
shared_nameAdam/dense_26/bias/v
y
(Adam/dense_26/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_26/bias/v*
_output_shapes
:
*
dtype0
?
Adam/dense_26/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*'
shared_nameAdam/dense_26/kernel/v
?
*Adam/dense_26/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_26/kernel/v*
_output_shapes

:
*
dtype0
?
Adam/dense_25/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_25/bias/v
y
(Adam/dense_25/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_25/bias/v*
_output_shapes
:*
dtype0
?
Adam/dense_25/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_25/kernel/v
?
*Adam/dense_25/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_25/kernel/v*
_output_shapes

:*
dtype0
?
Adam/dense_24/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_24/bias/v
y
(Adam/dense_24/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_24/bias/v*
_output_shapes
:*
dtype0
?
Adam/dense_24/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?N*'
shared_nameAdam/dense_24/kernel/v
?
*Adam/dense_24/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_24/kernel/v*
_output_shapes
:	?N*
dtype0
?
Adam/dense_27/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_27/bias/m
y
(Adam/dense_27/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_27/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_27/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*'
shared_nameAdam/dense_27/kernel/m
?
*Adam/dense_27/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_27/kernel/m*
_output_shapes

:
*
dtype0
?
Adam/dense_26/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*%
shared_nameAdam/dense_26/bias/m
y
(Adam/dense_26/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_26/bias/m*
_output_shapes
:
*
dtype0
?
Adam/dense_26/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*'
shared_nameAdam/dense_26/kernel/m
?
*Adam/dense_26/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_26/kernel/m*
_output_shapes

:
*
dtype0
?
Adam/dense_25/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_25/bias/m
y
(Adam/dense_25/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_25/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_25/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_25/kernel/m
?
*Adam/dense_25/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_25/kernel/m*
_output_shapes

:*
dtype0
?
Adam/dense_24/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_24/bias/m
y
(Adam/dense_24/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_24/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_24/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?N*'
shared_nameAdam/dense_24/kernel/m
?
*Adam/dense_24/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_24/kernel/m*
_output_shapes
:	?N*
dtype0
w
false_negativesVarHandleOp*
_output_shapes
: *
dtype0*
shape:?* 
shared_namefalse_negatives
p
#false_negatives/Read/ReadVariableOpReadVariableOpfalse_negatives*
_output_shapes	
:?*
dtype0
w
false_positivesVarHandleOp*
_output_shapes
: *
dtype0*
shape:?* 
shared_namefalse_positives
p
#false_positives/Read/ReadVariableOpReadVariableOpfalse_positives*
_output_shapes	
:?*
dtype0
u
true_negativesVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nametrue_negatives
n
"true_negatives/Read/ReadVariableOpReadVariableOptrue_negatives*
_output_shapes	
:?*
dtype0
u
true_positivesVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nametrue_positives
n
"true_positives/Read/ReadVariableOpReadVariableOptrue_positives*
_output_shapes	
:?*
dtype0
{
false_negatives_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_namefalse_negatives_1
t
%false_negatives_1/Read/ReadVariableOpReadVariableOpfalse_negatives_1*
_output_shapes	
:?*
dtype0
{
false_positives_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_namefalse_positives_1
t
%false_positives_1/Read/ReadVariableOpReadVariableOpfalse_positives_1*
_output_shapes	
:?*
dtype0
y
true_negatives_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:?*!
shared_nametrue_negatives_1
r
$true_negatives_1/Read/ReadVariableOpReadVariableOptrue_negatives_1*
_output_shapes	
:?*
dtype0
y
true_positives_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:?*!
shared_nametrue_positives_1
r
$true_positives_1/Read/ReadVariableOpReadVariableOptrue_positives_1*
_output_shapes	
:?*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
?
MutableHashTableMutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_nametable_2016626*
value_dtype0	
o

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name	2041754*
value_dtype0	
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
r
dense_27/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_27/bias
k
!dense_27/bias/Read/ReadVariableOpReadVariableOpdense_27/bias*
_output_shapes
:*
dtype0
z
dense_27/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
* 
shared_namedense_27/kernel
s
#dense_27/kernel/Read/ReadVariableOpReadVariableOpdense_27/kernel*
_output_shapes

:
*
dtype0
r
dense_26/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_26/bias
k
!dense_26/bias/Read/ReadVariableOpReadVariableOpdense_26/bias*
_output_shapes
:
*
dtype0
z
dense_26/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
* 
shared_namedense_26/kernel
s
#dense_26/kernel/Read/ReadVariableOpReadVariableOpdense_26/kernel*
_output_shapes

:
*
dtype0
r
dense_25/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_25/bias
k
!dense_25/bias/Read/ReadVariableOpReadVariableOpdense_25/bias*
_output_shapes
:*
dtype0
z
dense_25/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_25/kernel
s
#dense_25/kernel/Read/ReadVariableOpReadVariableOpdense_25/kernel*
_output_shapes

:*
dtype0
r
dense_24/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_24/bias
k
!dense_24/bias/Read/ReadVariableOpReadVariableOpdense_24/bias*
_output_shapes
:*
dtype0
{
dense_24/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?N* 
shared_namedense_24/kernel
t
#dense_24/kernel/Read/ReadVariableOpReadVariableOpdense_24/kernel*
_output_shapes
:	?N*
dtype0
G
ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R 
H
Const_1Const*
_output_shapes
: *
dtype0*
valueB B 
Q
Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 R	????????
I
Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 R 
??
Const_4Const*
_output_shapes	
:?N*
dtype0*??
value??B???NBiBtoBtheBaBmyBandByouBisBitBforBinBofBimBonBmeBsoBhaveBthatBbutBjustBwithBbeBatBitsBnotBwasBthisBnowBgoodBupBdayBgetBallBoutBlikeBareBnoBgoBdontBdoByourBtodayBtooBgoingBloveBworkBcantBgotBtimeBfromBbackBlolBuBwhatBoneBwillBknowBweBaboutBreallyBamBhadBcanBseeBsomeBwellBifBstillBwantBnewBnightBhowBthinkBampBthanksBhomeBasBwhenBthereBohB2BmoreBmuchBoffBmissBhereBtheyBneedBlastBanBthenBbeenBmorningBhopeBhasBgreatBtomorrowBillBtwitterBorBherBthatsBhahaBfeelBagainBsadBheBfunBwishBonlyBwhyBrightBdidntBsleepBbadBwouldBveryBhappyBsorryBbyBdidBtonightBcomeBmakeBthemBwereBgettingBwayBgonnaBthoughBniceBoverBiveBbetterBwatchingBshouldBsheBwaitBcouldByeahBbedBweekByoureBpeopleBschoolBhateBhimBdaysBevenBafterBheyBnextBdownBweekendByesBawesomeBneverBthankBsoonBtakeBlittleBlongBfirstBworkingBwannaBwhoBsayBbestBpleaseBdoingBhavingB4BbeingBshowBtiredBsickBwatchBeveryoneBhisBokBourBwontBlifeBanyB3BdoneBfeelingBalwaysBsureBfriendsBalreadyBthingBthanBanotherBusBfindBcoolB	somethingBguysBreadyBmadeBxBwhereBbecauseBlookingByayBwentBlookBmanBphoneBdoesntBurByetBhoursBbeforeBhouseBmovieBprettyBeverBtryingBawayBmaybeBomgBfinallyBoldBhelpBsummerBletBamazingBearlyBthingsBintoBleftBlostBtweetBguessBfollowBdamnBkeepBthoughtBsomeoneBbigBmissedBbitBlt3BsameBhotBhaventBwhileBsucksBnothingByearBrainBstartBfriendBgladBtryBwowBotherBcomingBalsoBtellBlooksBbirthdayBboredBliveBdoesBtwoB1BhearBgirlBlaterBweatherBactuallyBthoseBsawBbabyByaBsunBsongBisntBmakesBstuffBmightBexcitedBwaitingBnBpartyBsaidBhardBplayBhesBsinceBuntilBgameBfewBughBsuchBlotB	yesterdayBgottaBlateBaroundBgodBidBhiBmyselfBworldBmanyBcarBsoundsBfoundBmusicBluckBcheckBtheirBheadBjobBgiveB	beautifulBmustBfridayBreadBmakingBcoldBcallBwhatsBputBgoneBtalkBmayBsundayBmissingBawwBleastBanythingBwokeB5BpoorBtillBmomBstopBmondayBuseBleaveBmostBalmostBtimesB	listeningBokayBcuteBthoBhairBfarBwantedBhurtsBlunchBmeanBeatBiphoneBfreeBfamilyBtheresBenjoyBfunnyBshesBfoodBfinishedBhourBendBdinnerBbelieveBplayingBforwardBanyoneBwelcomeBwithoutB	followersBthinkingBshitB
everythingBsweetBwhichBcauseBtotallyBvideoBtheseBwasntBbuyBoutsideBenoughBstupidBhahahaBthroughBweeksBrBmineBcoffeeBwrongBeveryBrealBanymoreBprobablyBcouldntBwBplaceBonceBeatingBwantsBroomBstayBtweetsBmoneyBxxBbusyBsoooB	followingBtvBwinBhellBhaBlovelyBwholeBcameBseenBsaysBtakingBsaturdayBpicBkindaBkidsBclassBexamB10BbeachBbothBtookB	hopefullyByearsBcrazyBheadacheBletsBsuperBnameBideaBdBhelloBableBdadBnewsBtheyreBhalfBtrueBforgotBguyBbookBmeetB	goodnightBlotsBpostBawwwBleavingBfaceBsittingBgirlsBsendBrestBeitherBagoBusedBoBreadingBelseBfullBfeelsBshoppingBsooBhurtBcomputerBrunBseemsBahBtalkingBcuzBwatchedBrainingBownBtriedBhitBrememberBneedsBstuckBheardBaloneBblogB6BbBtripBfuckBofficeBbooBstartedBkindBdogBbtwBcourseBheartBseeingBheheBinternetBpartBmindBusingBquiteB
mileycyrusBonlineBpictureBaddBtoldBawakeBlovedBpicsBgoesBboyBfineBcryBtilBpainBbreakB	breakfastBlaBchangeBgetsBwakeBsBboughtBsunnyBpersonBboringB	seriouslyBbrokeBupdateBminutesBcareBcalledBfacebookByoullBstartingBconcertBseasonBopenBpayBluckyBreplyBasleepBawBdudeBlmaoBassBjuneBbringBfavoriteBlinkB8BhungryBcrapBsiteBheadingBanywayBinsteadBemailB30BsleepingBmothersBxdBwalkBtrainBmonthBstudyB	afternoonBdriveBshowerBfanBjealousBenjoyingBtommcflyBexamsBredBboutB100BtextB	wonderfulBmadB
definitelyBhopingBsoreBiceByeaBmoveBsooooB7BbyeBrunningBfinishBtogetherBproblemBrockBbdayBdiedBcongratsBmeansBaskBhighBonesBworksBhappenedBfuckingBdeadBgoinBfailBsisterBcityB	sometimesBhomeworkBcoupleBwriteBboysBdearBwonBmoviesBalbumBdrinkBcomesBsuckBcutBlaptopBlovesBwouldntBbrotherBpB12BtopBsetBmonthsBddlovatoByoutubeBeyesBtourBchurchBarentBpplBipodBreasonBsoundBhappenBwaterBteaBeveningBvisitBperfectBfinalBsongsBdreamBlilBtownBmeetingBlistenBstudyingBnapBweirdBseemBfallBniteByallBsighBlovingBsideBdanceBticketsBgymBcloseBtestBlessBhangBmoodBenglishBfbBateBinterestingBknewBcatchBcatBagreeBcreamBsecondBcleanBturnBlistBstoreBmomentBworstBaintBwritingBgtBstoryBsayingBawardsBwordBahhBrideBsupposedBworthBpoolB	chocolateBwishingBsmileBbrokenBlondonBfastBviaBunfortunatelyBpageBmovingBpastBdrivingBairB20BthreeBxxxB1stBthroatBforgetBsentBpicturesBgaveBdreamsByepBweddingBdaBshortB
understandBphotoBparkBcleaningBfollowfridayBsunshineBhorribleBblackBsleepyBdrinkingBpickBjonasBplanBtweetingBchanceBcollegeBaccountBstarBwonderBworseBratherBunderBlongerBfellBslowBteamBemBvoteBhugsBhmmBsatBcannotBscaredBbetBeasyB
apparentlyBparentsBdateByouveBupsetBdueBmoonBspentBflightBgreenBpointBladyBspecialBmacBhugeBholidayBupdatesBmtvBplansBmumBspendBhowsBtuesdayBhangingBhandBfluB9BplusBfairBnopeBearlierBjoinBcBthxB	wonderingBwtfBwordsBshowsBbandBmileyBduringBpsBshameBwebsiteBworryBlazyBbodyBbusBsleptBmessageBwearBukBbrothersBanswerBforeverBthinksBvacationByBwhiteBstomachBahhhBwarmBbeerBmrBlookedBjonasbrothersBfigureBlearnBvoiceBthursdayBsadlyBidkB
especiallyB	differentBsupportBfansBjulyBdieBcakeBmeantB15BlineBsimsBinsideBchatBmetBgoogleBitllBphotosBlikedBnumberBmyspaceB
officiallyBepisodeBfixBsafeBrainyBdavidBcameraBairportBcryingBdressBsmallBpizzaB
absolutelyByummyBbbqBtomBshopBtummyBgamesBshallBworkedBfeltBdecidedBluvBpaperBproudBripB	boyfriendB
graduationBeachBpowerBgardenBfinalsBprojectBsaveBexceptBshoesBneededBbeatBeyeB2dayBwitByourselfBkillBbikeBradioBplayedBroadBhugBgorgeousBstartsBlonelyBkeepsBnightsBannoyingBblueBbooksBappleBchickenBexactlyBhospitalBalrightBcaseBwishesBexcitingBcosBsignBkidBhatesByupBfrontBcardBtwilightBtakenBfeetBfrenchBlivingBwineBsonBfactB	wednesdayBdmBxoxoBnearBlameBbabeBturnedBpackingBwooBcdBlaughBpinkBknowsBgoodbyeBrealizedBscaryBhubbyB2ndBshareBhappensBclubBbehindBquestionBsoldBbcBouchBjusBdownloadBwakingBgettinBbusinessBpassBcupByoursBserviceBmomsBgivingBkillingBvideosBloseBwalkingByoBdrunkBminuteBclothesBorderBappBalthoughBalongBenjoyedBrelaxingBhahahBterribleBfilmB	everybodyBbroBalotBpassedBtouchBguitarBpostedBcompanyBhasntB	fantasticBsisBrandomBwhateverBvegasBrevisionBsitB11BversionBlakersBstayingBaskedBaheadBminsBroundB	interviewBsingleBindeedBboxBmotherBhmmmBsingingBlightBvipBbummedB
completelyBflyBcommentBuploadBhistoryBdyingBdarkBdealBwifeBohhBehB	currentlyBbetweenBlatelyBwebBreBfreakingBchangedBusuallyBwearingBshirtBplaneBlBbitchBhunBothersBhuhBheadedBdisappointedBbloodyBhangoverBworriedBawfulBwhosBshootBcampBfingersBpeaceBfmlBgoshBtrafficBranBwatchinBsingBquickBhotelBholdBspendingBcaughtB	tweetdeckBffBdeathBextraBbarBseriousBnearlyBcountryBbbBfathersBitunesB	somewhereBsexyBdogsBcozBfatBartB	exhaustedBclosedB
appreciateBpissedBmatterBfixedBprofileBdoorBhrsBcheeseBshouldntBmathsBnoseByumBchillinBcookBdunnoBbloodBtakesBproblemsBoohBdvdBpcBdefBmathBnaBendedBmommyBhandsBstoppedBdangBnycBmallBsillyBsoooooBticketBgroupBdaughterBawwwwBgayBinfoBnobodyBshotBnickBlowBfeverBmagicBfutureBwindowsBnoteBpuppyBkBfamBnoneBfitBselfBtypeBswimmingBtaylorBscreenBplanningBlayingByoungBclassesBwindowBbagBtroubleBplzBpossibleB
blackberryBpracticeB	starbucksBfireBdoinBheatBtrainingBdropBcallingBcongratulationsBblessBoftenBconfusedBwatBnahB	hilariousBmateBtodaysBmiddleBnormalBdancingBprayBarghBthruBteethBsortBblahBfailedB
backgroundBdeBstormBrecordBquietBnyBdaddyBjonBthoughtsBtanBwetBetcBmajorBburntBseriesBmcflyBlegsBbummerBstreetBshutBpeepsBmBminBhusbandBmentionBfavBmailBdroppedBhillsBwomanBtrekBmaBputtingBsendingBquottheBfightBeBdemiBdoubtBusualBjoeBstraightBcousinB♫BpmBfootBbadlyBmmmBeventBkateBtrailerBsupposeBbrainBchoiceBimaB	availableBchrisBbankBdoubleBmilesBguttedBfranceBspeakBpaidBepicBtasteBstandBquotBhelpingBdavidarchieBjohnBjkBlearningBlaundryBsistersBkickB	cancelledBsouthBvBcheckingBfishBhahahahaBfreshBfallingBloadsBstateBdadsB	australiaBprobBideasBsaleBbuddyBsearchBfourBtwittersBagesBhoneyBfeelinBchillBafraidBpackBdoctorBsmellBcoverBanywaysBaskingB	depressedBkeepingBmsnBdonniewahlbergByrBmilkBmenBchicagoBlikesBkilledBmovedBkiddingBdntB24BpromBcriedBchannelBbecomeBacheBwheresBwootBbuyingB
twitteringBnervousBsenseBdentistBlegBfindingBknoBskyBsanBmouthBcompleteBblastBarrivedBdarnBlosingByoudBswineB50BtrackBtellingBamericanBfreakinBbunchBeverydayBb4BcheckedBballBmissesBladiesBcheersBholyBvsBrocksBmessBjonathanrknightBraceBadamB3rdBtweepsBthanxB	happeningBagainstBtBupdatedBbfBanywhereBxoBcallsBshoutBaffordBweveBsellBpromiseBmobileBouttaBgigBcheerBhoweverBgottenB	expensiveBcellBmidnightB
depressingB	favouriteBareaBhannahBunlessBdesignBfollowedBcatsBshowingBfiveBcrappyB
experienceBallowedBbroughtBlakeBgermanyB	deliciousBwroteBfolksBwantingBuhBsomebodyBaddedB	importantBplsBstayedBgunnaBtalkedBcookiesB25BcopyBkittyBbrazilBenergyBweekendsBaugustBtwiceBloadBreturnBstickBcookingBjokeBmessagesBissuesBrelaxBimagineBjoyBquotiBspaceBcountBbrightBwoohooBtoniteBemptyBsushiBuniBrollBpositiveBcousinsBrowBworkoutBstBsometimeBcodeBscienceBfloorBdrBstudioBwerentBshiftBjordanknightBbathBfaveBdannyBenglandBlibraryB	excellentBmatchBfinaleBsexBentireBdisneyBpickedBneckBnoticedBstationBadorableBviewBbossBheresBlocalBooohBsessionBglassBsocialBkissBsweetieBearBoopsBimmaBdrinksB2009BgiftBtearsBreleaseBkeptBtoughBbatteryBcatchingBofficialB40ByorkBringBwiiBooBthunderBmitchelmussoBtravelBinviteBislandBvidBxboxBloudBsoupBpersonalBsecretBdeserveBbarelyB18BfollowerBcoughBboardBstressB
everywhereBturnsBsystemBmarkBmamaBsumBgradBlieBburnBsimpleBnailsBfabulousBaliveBthrowBswearBlaughingBageBcouchBwestBpieceBlettingBkitchenBhurtingBbugBbuttonBpantsBarmBholidaysBperhapsBparisBohhhBflyingBtrustB
productiveB	brilliantBtweetedBpopBmachineBarticleB♥BhavBdammitBstrongB13BmorninBcanadaBjbBsharingBkneeBchineseBsurpriseBamericaBdramaBbgtBeasierBbuttBumBharryBrealizeBgoodnessBuglyBfingerBcommentsBbreakingBsquarespaceBannoyedBlessonB14BsomehowBbillBdcBtaylorswift13BlossBcravingBlayBendingBstrangeBbffBmarathonBrobBmediaBlandBstepBfiguredBtexasBahhhhBatmBboatBreportBstageBwedBstyleBessayBblockBdietBplannedBhahBreviewBhelpsBgivenBmarriedBacrossBglassesBfeedB	extremelyBpresentBevilBdumbBbirdBsirBplacesBthaBaddictedBmsBspanishBresultsBmessedBgahB	apartmentBanytimeBspotB16BhehBjackBflatBdeskBbottleBmanagedBsaBapartBfootballBspamBfakeBdryBjamesB
interestedBsurgeryBsoccerBpossiblyBnoooBlookinB17BteacherB	difficultBampampBtotalBorderedBweightBwallBpleasureBhooB	finishingBspringBsignedBheheheBshiningBgivesBroughBscheduleBfestivalBbrownBrevisingBpriceBnorthBvotedBminiBemailsBruinedBbabiesBworriesBswimBlackBfloridaB4thBsizeBontoBltBdegreesBworkinB	everyonesBe3BfaultBturningBsixBtrulyBpublicBofferBosBwashBangryBdailyBbostonBlivesBhearingBpostingB
connectionBmemoriesBjuiceBsamB	questionsBalexBnoticeBilyBissueBbreadBwinterBtwitBshittyBgrrrBwomenBlatestBclickBtalentBrealityBmarketBwalkedBblockedBndBdecideBlearnedBlivedBtrendingBmostlyBserverBleavesBblameBmattBdirectB3dBwasteBorangeBmixBblessedBmoBchillingBabtBfuckedB	celebrateBnastyBlmfaoBadviceB21BlordBjoinedB
girlfriendBfuckinBshotsBpulledBbiggestBcolorBahaB2morrowBmondaysBexpectBcheapBclearBawhileBnooooBepisodesBmiBskinBdatBcrossedBcrashBsunburnBgrandmaBfabBdoctorsBchildBaccessBacBheavyBremindsB	otherwiseBmacbookBproBmillionBlawBgrowBupdatingBcaliB	diversityBughhBhealthyBsmartBcoastBmichaelB	attentionBanybodyBslightlyBnetBtreeBpainfulBquitBappsBearthBlinesBdragBprayingBcreditBheckBgoodmorningBcloserBchildrenBlinksBasapBridBcontrolBtwitpicBaccidentB	obviouslyBhopesB
conferenceB	beginningBplayerBkingBhellaBmassiveBthtB	literallyBdirtyBmmmmBalarmBweeBsaladBgrandB	everytimeBbaseballBstatusBpaintingBgolfBkickedBdeletedBknowingBphonesBconanBwwwtweeteraddercomBsusanBquoteBupgradeBtoastBpagesBearsBjamBcableBbearBtheyllBolderBdeepBtableBmontanaBcompBformBeatenBsugarBmikeBlevelBhandleBcontestBactingBtransformersBthirdB	surprisedBgrossBcardsBwifiBsarahBheroBbuildingByardBtennisBpairBlovinBgtltBandyBtonsBlikingBfreezingBclientBrepliesBpreferBthankyouBcrashedB
californiaBtopicBopeningBhurryBryanBselenagomezBsooooooBcontactBsydneyBdougiemcflyBjBtxtBwingsBquicklyBprayersBpostsBflowersBfellowBsnowBfamousBneitherBfedBarmsBpullBowBjesusBoriginalBdaveBfightingBstressedBrecentlyBkeyB2niteBcountingBsoftwareBsmhBqualityBtenBshellBhardlyBtyBtinyBpackedBlimitBknownBsundaysBpaintBhatBgermanBjobsBhillBeastBcostBangelsBtixBperformanceBwideBrealisedBgrrBvisitingBaweBendsBeuropeBactB
rememberedBproperlyBankleBsuckedBexcuseBslowlyBcloudyBbirdsBnetworkBridingBpaBtearBgradeBcarsB	uploadingBjumpBgasB140BmemoryBdecentBtoothBdowntownBbaconBahahaBjayBdownloadingBchooseBcashBpayingBmealBhttptweetsgBteachBretweetBhealthBhavinBhaircutBmainBpickingBkevinBcertainBcancerBanniversaryBsuxBatlBwashingBsportsB	preparingBinvitedBtestingBriceBadBtruckBprogramBpointsBmehBbrandBbenBsuccessBhelpedBchipsB
assignmentB	allergiesBuncleBnamesBidolBhonestlyBftwBcenterBcavsBwindBpresentationBdetailsBproperBlyingBgrowingB	certainlyB45BexpectedBtextingB	impressedBstarsBsoulBbayBscrewedBdressedBriteBreceivedBmakeupBbiggerBactionBsandwichBeggsBtryinBsmokeByahBwheneverBtipBashleyBwarsBxxxxBvetBpeterfacinelliBcontinueBwinningBtattooBstoriesBstokedBfB200BrubbishBmissinBgB3gBsmellsBaimBagreedByehBpetBboooBscoreBsavedBhadntBlaurenByrsBsmilingBfasterBbritneyBpotterBewBreachBebayB�BhittingByouuBusaBreviseBattackBsuggestionsBmiamiBwisdomB0BjeansBtruthBbornBmuseumB
dannymcflyBspeakingBpreBgnightBorlandoBmaxBtestsBfilledBbesidesBtechBlikelyBbetaB	miserableBauntBwaysBhwBguiltyBbedtimeB
restaurantBinstallBseaBnotesBnationalBerBclassicBtmrwBexplainBeditingBdallasBbeginBfatherBchattingBseatB	recordingBthemeBsitesBpurpleBpancakesBburnedBsceneBresearchBhumanBbutterB1000BsecondsBfearBcurrentBcandyBbathroomBsigningBkoBseattleBvotingBangelBcampingBoddBedBtightBbiteBstudentBpubBcokeBtoeBitalyBtweetersBgossipBatleastBpaulBharderBbottomBreleasedBphilippinesBmodeBiranBthrewBheyyBheavenBfashionBcornerBchickBchargeBwithinBughhhBoptionBwwwtweeterfollowcomBtshirtBgr8B	recommendBavatarBwickedBtipsB
terminatorBstoleBchangingBconsideringBspainBkeyboardBtorontoBerrorBcrewB	basicallyBwoBwhoaBtweetieBheeBfancyB	situationBseveralBpartiesBcrackBalcoholBcreativeB5amBwaveBtreatBgreatestBwarBsettingBrepliedBnowhereBlogBhmBskypeBcuriousBchuckBakaBadmitBzooBswitchBparadeBshakeBkissesBjerseyBprincessBfirefoxBwouldveBwoopBmodelBjasonBdeleteBdataBummB	tomorrowsBcuddleBcloudsBsuddenlyBspellB6amBrunsBmorningsBlightsBiiBelBbellyBwoooBmigraineB
differenceBcaBstandingBjoshBgroceryBapluskBplentyBperBpeoplesBcoveredBwokenBshowedBremindB	nightmareBcrossBhayfeverB
strawberryBbookedBwhetherBtoooBstarvingBmumsBmassageBloserBheadsBcookieBappointmentBwastedBgoalBfocusBwinsBdroveBwenBsteveBrainedBflashBfeelingsBexBballsBregularBpalmBlockedBholeBbeyondBspreadBspeedBfieldBbahB5thB
downloadedBtalkinBprobsBprivateBmistakeBf1B22BinsaneBholdingBdocBbunnyBtommorowBoooBdelayedBboyleBmedsBgreyBwildBresultBgoldBnamedBhatingBtaBswollenBhorseBconversationB	searchingB	happinessBclueBbananaBjoeymcintyreBimageBpooBnkotbBbringsBwrittenBshouldveBchairBcelebratingBtoyBqueenBpressBnothinBtuneBrudeBgeorgeBburningBbugsBbringingBdatesBpieBcafeBprocessBlaunchBitselfBclockB
ridiculousBnooneBhonestBbeautyBsomedayBattendBappreciatedBtermBcleanedBboomBrequestBthinkinBhungBcyrusBriverBfruitBbuildBwhilstBschoolsB	chemistryBactualBtheyveBperiodBkeysBgooodBnutsBlistenedBhrBtextsBshaundivineyBnooBepBbandsBanimalBletterBawardBsellingB	infectionB
frustratedB
charactersBchangesBpartsBleadBbowlingBshortsB
discoveredBclosingBbowlByogaBsfBresponseB
impossibleBscrewBhdBcupcakesBaddressBjustinBgrassBswiftBshoulderBdjB4amBvidsBpouringBlargeBidiotBchargerB500BstatesBrulesBpushBmeeBcominBbreatheBurghBtimB	mcdonaldsB	sufferingBseniorBmanageBjapaneseB	characterB23BtisBpicnicBbioBalByellowBoceanBinsomniaBfrmBdanB09BkickingBinspirationBalasBpianoBpeterBgeneralBgrabBbestieByuckBpityBdreadingBarriveB19BstreamBsavingBrespondBlooseBhimselfBolBcruiseBcinemaBgagaBfullyBdarlingBshootingBmexicoBcontentByoungqB	mentionedBlanguageBdrawingBbecomingBthnxBrepeatBlessonsBchestBroflBqueByouquotB	wolverineBtacoBexerciseBcanceledBparkingBlabBbellBhayBamyByuByeyBtonyBseatsBrushBkillerB	includingBdrawBdrankBcreepyB
basketballB7amB35BclearlyBbabysittingBattemptB3amB
incredibleBgraduateBmrsBmouseBgarageB	marketingBlyricsB	septemberBpleasedBirelandBfillBbotheredBaddingBwalmartBkatieBblowBstormsBprollyBcorrectBawhBgeekBrainsBkellyBcrushBawsomeBtagBsimplyBnailBstudentsBsadnessBpureBbreaksBomfgBhungoverBgfBgdBeggBcontactsBcerealBbluesBvanBsniffBskoolBprojectsBnephewBchelseaBbtBjessBjapanBgfalcone601BcntBmissionBkrisBaswellBrentBreasonsBdesktopBcastBbreathBpensBnakedBmagazineBgiantBbotherBtitleBtopicsBcouldveB	challengeBpassingBmusicalBfoxBdecisionBcreateBamountBdutyB90B60BuploadedBfuneralBsectionBnoonBmeetingsBitalianBfarrahB2moroBrobertBnieceBbombBsittinBhoustonBtomorowBjourneyBdemonsBureBmileBlipBignoreBdespiteBbagsBrichBmetroB	installedBbrunchBtargetBnumbersBjordanBhoBfridaysB	confusingBchapterBthrowingBpokerBobsessedBlolzBerrandsB3gsBsomeonesBiranelectionBwanaBps3B
manchesterBruinBmiaBbasedBwinnerBfilesB80BstockBmakinBfaithBcuttingBchinaBburgerBpapersBlawnBjimmyBitdBcaresBbingBbillsBmsgBshirtsBshineBsauceBdisBcoughingBaptBperezhiltonBfriedBconnectBbucksBaustinBriseBrelatedB	melbourneBlongestBicecreamBhawaiiBeasilyBthBsortedBphysicsB	forgottenBfileBsurelyBindiaBbooooBscratchBprideBnovemberB	desperateBtypingBsalesBfridgeBavoidBcompetitionBaprilBallergicBdeckBruleBlt33BbeginsB300BjokesBherselfBgearBconBfreakB	communityBbkBfreedomBfeedbackBenterBcutestBbritishBbeeBwondersBweakBregretBopenedB	lightningBgrownBeventsBawwwwwBsebdayBmountainBmaryBgloriousBdisappointingBcarefulB	accordingBminesB
apprenticeBangByeahhBseemedBnoooooBfriendlyBconsiderBsurviveBsimilarBquotyouBnormallyBkittenBivBbumBquizB	expectingBatlantaBtalentedBmonBmexicanBfeatureB
commercialByikesBvirusBsunburntBoilBngBenBsuitBphillyBofflineBhighlyBgloomyBhideBeuB360BtwitterverseB	tomfeltonBplayinBlvattBilBcommonBskillsBkingsBhockeyBheelsBnomBmommaBmeaningBfestBttBmatesB	hollywoodBdishesBtonBpastaBdriverBdiBbakingB8amBtonightsBprintBrateBsoftballBgroundBcancelBscotlandBreplyingBmomentsBflickrBdvdsBcomfyB	screamingBbobB	saturdaysBlandedBupsBteachingBchipBbedroomB	adventureBwoahBtastesBmeatBcolourB6thBoutfitBnonBmemberBkimBfrustratingBfriBusefulBericBanimalsBrockedBpeanutBleBhipBespBdegreeBdamnitBtellsBmonkeyBhahahahBallowBgonaBforumBforceBpreviousBjstBgoldenBsingerBpoorlyBoctoberBcourtBcentralBtongueBtanningBeffectB	singaporeB	neighborsBfailingBdamBcookedBunfairB	rehearsalBforcedBterriblyBpotBtastyBspiderBmornBfriesBthursBsoonerBnerdBmonsterBjacketBieBhookBdsBcustomerBsryBopinionBmahBgrrrrBdependsBblehBaceBuniverseBstoresBpartyingBcamBbeinBaccountsBpopularBnoiseBcrowdBvaBpressureBguessingB2010BspiritBnightyB	microsoftBleeBlaughedBuserBsquareBsmokingBgeniusB
eventuallyBaccidentallyB
technologyBspeechB	photoshopBofficialtilaBdollarsBcomedyBspokeBpackageBluckilyBhavntBcmonBcabBblinkB530BqBmassB
incrediblyBroofBpopcornBcrashingBcoBassignmentsBwindyBtrynaBstaffBplaysBjeffBinvolvedBinspiredByahooBthatllBhoorayBghostBedwardBdriversBbrosBitchyBdiegoBaboveBtgifBbattleBartistBsubwayBpushingBemilyBcricketB	christmasBtunesBstolenBshoeBrtBmequotBlifesBlangBfacesBbritainsBatamptBfixingB	computersBblogsBwalletBsoundedBprinceB
graduatingBdawnBbootsBbitesBurlBstealBsecurityBpoliceBkirstiealleyBdeliveryBcalmBamazonB	wordpressBsocksBsiBhanginBgudB
courseworkBblondeBtheatreBrealiseBphotographyBperformBacceptB21stBrespectBparanoidBfemaleB
disneylandBbbcBwaitinBmeeeBbitchesBwirelessBgodsBtoesBtbhBpodcastBhuntingB
headphonesBtakinBsnapBsmilesBroseBdunBcarryBsuddenB
recoveringBcomicBbullBandyclemmensenBtheaterBdareBvodkaBremindedBcomplainBbiologyBaudioB8thBwaitedBsteakBspareBmixedBhostBhidingBgaBthankfulBmusicmondayBishBentertainingB
collectionB80sBshipBshinyBsevenBprogressBmanagerBhBsomethinBprepareBironBflipBevery1BbassB2amByouuuB
universityBtheoryBguestBbrianBbonesBandrewBselenaBsayinBreunionBnbaBalexalltimelowB
stephenfryBhonBhitsBexistBdreamingBtowardsBclientsBamandaBstoneBhvBgmailByerBnoodlesBnicelyBnatureBhookedBbeatsBapplicationBuselessB
sunglassesBrubB
motivationBlayoutBknightBahhhhhByesssBpasswordB630ByayyBummmBtwinBshedBscriptBpoopBplanetBfyiBdonutsB	celebrityB	emotionalBdroppingBbillyBapptBtypicalBpiecesBmembersBkobeBfishingBeffingBcureBthingyB
successfulBremovedBlicenseBturkeyBjetBfrozenBbabesBseBi´mBenvyBcherryBbitsBasianBstuffedBmaintenanceBicedB	headachesBdeservedBsuggestBohhhhB	graduatedBeditBdigitalBstinksBsoftBplainBoursBmedicineBbakBayeBwashedB
sweetheartBshapeB	completedBcdsBbustedBawkwardB
afterwardsBwossyBwarpedB
tracecyrusBshopsBlogoBfoBcreatedBbubbleBsemesterBgrandpaBflBcharlieB18thBusersBullBtwitsBrelationshipBhomemadeBgniteB
definatelyBrockinBpreviewBiamdiddyBoperaBlocationBhopBconcertsByouthBmmmmmB	insuranceBforgiveBcontractBbootBbeansBwkBpoundsBpornBpandaBhuntBbridgeBboxesBwahBskipBrachelBlt333BlisaBinformationBhmmmmBdrewBwthBtheeBxpBtapeBhallBdiesBcoolerBzeroBrayBgraceBfailureBtiringBoooohBinterestBgreekBfoolBtmrBshoutoutB	perfectlyBknockedBbellaBhiyaBgrandparentsBdisappearedBtankBnadalBjoiningBiconBwarningBreachedBmeganBjoeyBhatedBdullBcloudBtradeBrollingBreplaceBlemmeBgeezBashleytisdaleB26BpartnerBlilyroseallenBdemoBcupcakeBwiseBtwitterlandBopportunityBmmBlauraBeffortBbangByoungerBwudBrecipeBpillowBnorB	addictionByessBwrkBstairsBsoooooooBreceiveBohioBdollBcutieBpanicBloungeBhappierBcrampsBclipBblowsBwishedBsoloBprayerBpeeBintenseBallenBtheydBsettleBpromisedBpissBpatheticBheartsBachingByawnBunfollowBtoolBteenBstrawberriesBsouljaboytellemBmummyBindianBeasterBsupernaturalBpocketBmetalB
johncmayerB	dannywoodBbloggingB
personallyBnaturalBknockBexpressBdenverBcakesBtimingBshortlyBcarpetBbsbBtreesBthaiBsubjectBphewBoweBmasterBkyleBflightsBbuggerBahahBusernameBtoiletBsneakBsangBreviewsBpumpedBfilmingB	daughtersByayyyBscreamBplantBobamaB
nightmaresBlebronBgreysBdinBworeB	travelingBtracksBpreparedBlemonBwavesBtuesB
themselvesB
downstairsBbrowserBputsBmichelleBdickBarmyBanswersBspellingBsnackBnxtBnicoleBminusBignoringBcurryB	confirmedBchBboundBsantaBreturnedBpeBjonesBemmaBdigBbudgetBbuddiesBunBthunderstormBtaxBstopsBsplitBpeteBjunkBchoresBwkndBpricesBfillingBfawcettBellenBdealingBsuckyBkillsBexactB430BzoneBrawBproductBcoldplayBcoachBcareerBbsBroleBrockingBincludedBentryBbakedB	argentinaBappearBsonsBrecommendationBpuppiesBjimBjenBclimbB
strugglingBpullingBposterBfurtherBwillingBthumbsBpremiereBnorwayBlaidB	geographyBwaBvistaBunableBroastBrecentBovenBnoesBheroesBeatinBasot400BadsBworldsBwhooBlipsBeminemBapplyBtxBlowerBkristenBhollaBplaylistBplantsBjerkBjakeBfrigginBbackupB9amBpunchBkimkardashianBhonorBhelpfulBdublinBcusBtweepleBjazzBflewBcriesBcoolestBtooooBthunderstormsBtherealjordinBsmithBpregnantBloBexchangeBcomplicatedBbecameB730BwhoeverBunitedBtrickBthomasBfamiliesBcomplainingBbrowniesBuponBtalksBsafariBpigBparamoreBmarryBgratefulBcolorsBchemBtireB	sleepoverBscottBimagesBgrandmasB
forgettingB
excitementBdatingBstaysBsocietyBnawBmuscleBlenoBirishBgrumpyBgrillBfoodsBewwBescapeBemoBedgeBcaffeineBannaBwristBwhewBtehBstaringBpreciousBpixBnegativeBmentalBgotsBglasgowBfilmsBduhB27BmessingBloverBitemsBeffBdopeB
songzyuuupBshouldaBmagBgoooodBcruelBbleedingB
dannygokeyBcrackedBblindBappearsBachesBtitanicBpapaBfarmB
complimentBneBtwitterberryBteB
neighboursBhackedBguideBdollarB7thBwatsBthatdBpainsB
markhoppusBmarioBhpBansweredB400BvirtualBteachersBsonyBpretendBnuggetsB	nevermindBmultipleBmarleyBhomieBhedBheapsBdelayBdanielB
attemptingBrescueBcomfortB	stressfulBsomewhatBrlyB
installingBemailedBborrowBttylBsmsBrecitalBnjBlaurenconradBhorrorBhintB
hahahahahaBbrideB70BtemptedBskinnyBshockedBshockBresistBpillsBlinuxBbudBxoxBscareBremixBakoBrestingBncBincludeBforecastBblewBbarsBtunedBtallB	spymasterBshudB	petewentzBhoodBdudesBvotesBswingBsourceBmindsBkindsBclosetBsurvivedBsonicBrandomlyBozBo2BkayBizBhardcoreBeverythingsBwhoopB	vancouverBvampireBunhappyB
thankfullyBsleepsBsilverBsightBgmBbracesB830BseanBpurchaseBlisteninBjamieBcoverageBapBsunsBsortaBsignalBquotimBjohnnyBinternationalBgtgtB	christianBthumbBstephenB	shatteredBoptionsB
disgustingBafricaBubuntuBstephBroveBrecordedBrainbowBleagueBhostingBhigherBfollowsBbakeBarseBamenByogurtBsoakedBsnlBlegalBlapBkatyBjumpingBdmsB
cheesecakeBcapsBwhoreBtubeBteeBsummersBrobinBreportsBlilyBfedererB	emergencyBdizzyBaccentBsolutionBsafelyBaussieBswedenBphoenixBdrugsBconvinceBbeersBtieBpotatoBphilBmaleBindoorsBhsBalyssamilanoBadoreBvinesBstoppingBsneezingBsaltBreminderBreactionBplugBplasticBhollyBhardestBeffinBdressesBdislikeBvalleyBtwinsBtrashBpleasantBianBdvrBwheelBsweatBoprahBikeaB28BzombieBwornBusbBtweetupBsurfingBsportBshaunBreplacedBmariahcareyBkarmaBheldBcomB30secondstomarsBtherapyBsobBnokiaBhaloBhahaaBextremeB	clevelandBauditionB	answeringB	addictiveB2nightB15thBtwittervilleBsrslyB
quotupquotB	favoritesB	customersBcleverBbestiesBashBumbrellaBtrentreznorBremoteBjacobB
distractedBdiffBbackyardBvanillaB
twitterersBthouBsofaBsicknessB	overnightBmegaBlasBknwBinboxBconcentrateBtadBpositionBlakerByelyahwilliamsBsoberBshippingBservicesBpopsBooooBoclockBnzBlimitedBleavinBjudgeBiamBdohBchillyBxxxxxBuberBsettingsBrouteBpitBnineBadultBtacosB	statementBsimonB	receptionB	potentialB
performingBpatrickBntBneighborB
managementBjuniorBflagsBdougieBconvoBcentreBactiveBaccomplishedBtextedBtempBproductsBpenBmidBmeeeeBmarchBkoreanBjokingBgrewBdustBdistanceBrecoveryBjacksonBhumidB	furnitureBfoneBfailsBdyeBduckBdetroitBcreatingB	countdownBcelebrationBbeefBsweetsBmarksBfunctionBetsyBcowBcostsBboredomB2000ByuckyB	sunburnedBsdBremoveB	nashvilleBmovesBmemorialBliedBheadinBhalfwayBfeaturesBfavorBbizByeBtwasBrealyBlosBgirlyB	childhoodBcampusBbelongBalbumsB10amBsoxBnooooooBlockBfactoryBcurseBwrapBtapBplayersBjessicaBewwwBaaronBupcomingB
soundtrackBquotesBheyyyBgrindBaliceB17thBupsideBroutineBrepairBreckonBpraiseBpassionBmanilaBliftBentertainmentBeditionBddB	blessingsB
yesterdaysBwhoseBsausageBjrBdoorsB	deliveredB930BtripsBprBnicerBeveBdiaBcornBblipB95B6pmBwifeyBpjsB
photoshootB	pattinsonBpassesBmitchelBlaughsBfallenBbustB1amB	wimbledonBvictoriaBtaughtBstackBsolidBsaraBsandB
eurovisionBazBashamedBstudiesBrootB
refreshingBmp3BmintB	irememberBgirlieBdonnieB
tournamentBthusBnaughtyBmattersBloopBgigsBchromeBawesomenessBsodaB	shouldersBretardedBoppositeBmenuBi�mBdessertBcheeringBcastleBberlinBbasementB34BtweeterBrollsBpresentsBplateBownerBlegendBkicksBdomainBdisplayB
bestfriendBstoodBloveeBkneesBkaBflowerBbuiltBbradiewebbstackBbibleBanatomyB1030BtomatoBswayBsmoothBseasonsBmessyBkitBgunBfaBdevelopmentBcelebsBbonusBadvanceBsurprisinglyB	katyperryBjumpedBfliesBcomfortableBsyncB	suspendedBsetsBpurseBmichiganBdonB
washingtonBsilenceBringsBnecklaceBgurlBesBdivorceBdecemberB	amsterdamBacceptedBworkshopBsufferBrippedBrareBrangeB
earthquakeBdevilBchileBchaseB9thBsimBsheetsBproofBplurkBmmsBjoB	houseworkBfantasyB	decisionsB	charlotteBattackedBaaahBwkendBtoysBtheellenshowBstepsBrelaxedB	liverpoolBjackalltimelowBhenceB	greetingsBdissapointedBdeservesB19thB10thBultimateBsakeBprizeBneatBlizBjavaBearnBcrashesBcomparedBchiBtortureBtetrisBrioBrecoverB
interviewsBgroundedBbuenoBbrbBbeanB16thBvisitedBtouchedBsalonBpatientBoverallBliesBjellyBhornyBhandsomeBfunniestBcheaperB
volleyballBviewsBtragicBspoiledBsortingBmarriageBloadingBjailBgoodluckBelectricBdeadlineBphysicalBgingerBgeeBblanketBbeatlesBusesBtumblrBpoemBmonitorBlatteBlastedBholsBbraceletB10pmBtidyBstrengthBprisonBitquotBhopedBglobalBgimmeBfallsBeffectsBdocsB
constantlyB	connectedBchefBcapBbeyonceBautoBacousticBwooooBwebsitesBvirginBsytycdBpromoBmeiBironingBhottBengineBabsoluteBworryingB
suggestionBstunningBportBpaintedBnurseBhousesBgangBdooBcopBtendBmirrorBgeorgiaB	continuesBbuzzBbeatingB20thBpeacefulB	inspiringBgoodniteBcomboBblissBbearsB	amazinglyBaiB230ByaaayBweeklyBreaderBracingBnamanBdorkBcrossingBchannelsBbonBbelovedBbalconyBanxiousB	announcedBaahB99BspaBsecBridesBrefuseBportlandBhecticBdemandBblownB
australianB9pmB75BtubBserversBryanseacrestBreplacementBjoggingBhatersBftBdamageBausB	advantageBpropsBpissingBniggaBhumorB
housewivesBgainBchoseBbinBbestestBbabysBassholeBabitBxmenB	thereforeBsignsBscrubsB	scheduledB	referenceBpatienceB
myweaknessBdropsBarticlesBactorBvalueB
sunbathingBljBjesseBgooooodB	dangerousBctBcanadianBboooooBamazedBurgeB
supportingBstreetsBsourBmightyBmcB
impressiveBgamingBbackstreetboysB90sBtomozBtaskBstalkerBpolishBperryBmangoBlevelsBkaraokeBhughBggBfindsBcoversBcountyBcbaBwoodsBuntillBsunsetBpupBneighborhoodBmuffinBkittensBgrayBdollsBcrushedBburgersBbastardBbannedB	attendingBwallsBuhhBsprintBrugbyBraiseBpepperBkatBjeezBgonBbradBbeastBbalanceBappliedB5pmBvenueBughhhhBtonysB	tommorrowBpurposeBpassportBjoysBinjuredBeekBeclipseBdesperatelyBcudBcharityBboothBbillyraycyrusBberryBapplesB330B32BzacBtrainsBordersBinstantBhazB	gardeningBfamiliarB	entertainBderbyByousBvibesBpatBlettersBdoggyBdiscBdhBbaseBasksBtskBstatsBrichardBocB
nickcarterBmusclesBmtBmelBhuluBhorsesBgraphicBfairlyBdrinkinBdenBchancesBbaB130BurbanBsurfBregisterBpoppedBpileBmontrealBlionsBexampleBcrankyBcombinationBallergyBverizonBsouthernBsmoothieBsinusBseniorsBsaddestBrabbitBmarsiscomingBlastfmBgroupsBdisasterBbrandonBaghBwotBwastingB	streamingBspinningBspinBrossBneedingBhaiBclosestBbullshitBassumeBweathersBreliefBkillersBg1BforumsBfootyBeconomyBblowingBblankBaveBspotifyBspilledBsome1BsatsBrarelyBpatchBfiguringBdonutBcommercialsB	apologizeB31BwebcamBtigerBtaxiBstrikeB	motivatedBeightBslightBsighsBshakingBsessionsBschofeBridiculouslyBramBprofessionalBparadiseBfitsBcalendarB
adventuresBabcB
twitterfonBtellyBstalkingBspicyBskiesBroommateBkoreaBhiccupsB	edinburghBchampionshipB	bamboozleBveganBthrilledBsetupBnwBlololBgraphicsBawareBaverageBthinBswitchedBskirtBignoredBcycleBcustomBcrimeB13thByessssBpetsBnintendoBninjaBnewestBitouchBimmediatelyBickyBgiftsBebassmanBdutchB	convincedBcelebBbreezeB	breathingBboneBauntsBwierdBporkBperthBlegitBgarlicBfeedingBdrumsBdelB	concernedBchocBchainBbrooklynBwilBtoolsBtatBpspBlensB
internshipBinkBhandyB	generallyB
friendshipBdamnedBcrystalBbeesBartistsBwaffleBsupperBshanedawsonB
networkingBnapsBmartinBgnaBfitnessBdrivesBdishBcribBbabygirlparisBapiBwalesBtermsB	stressingBroyalBpushedBmandyBcoloradoBcoatBwwwiamsoannoyedcomBsubBspearsBshyBrecommendationsBno1BlukeBfreakedBfrankBchicksB2pmBworshipBwhensBupstairsBtylerBsmileyBquotnoB
particularB	paperworkBlayinBdeviceBdeliverBconceptBbnpBbeliveB	archuletaB29B	twitteredBthrownBteamsBsticksBstalkBrafaBobviousBmuseBmariaBlovatoB	knowledgeBhumidityBgregBgaryBcsiBcopsB8pmBwritersBwoodB
unexpectedBtentBsoapB	remindingBpunkBpanB	newcastleBmicBlushB	internetsBhyperBgrilledBetBbashBamusingB3pmByouuuuBwidBtweetiesB
summertimeBspeakersBrubyBroomsBradBowlBovercastBotBofferedBmandyyjirouxxBit´sBgirlfriendsBgainedBfryBfarmersB	directionBcheesyBbuttonsBbumpBbowBbesideBtrendBtoooooB	sooooooooBshiftsBrecordsBquotaBptBpoliticsBoocBloggedBlobsterBkhloekardashianBhireBhamB
discussionBcutsBwwdcBtransferBsweatyBshoweredBrapB	pointlessB
mrskutcherB	knackeredBiphonesBhabitBgmorningBcodingBwutBuncomfortableBthreadBstrangerBromanticBrogerB	presidentBprepBpearlBmuffinsBmadnessBlitBimproveBgiveawayBdevBabilityBwksBwinkBthousandBteaseBsmashedBshaneBribsBnovelBlouisBdeanBbraveB1130BwhtBwafflesBsevereBpenguinsBnutBnpBnetflixBloooveBjaneBgreetBfunnB	franciscoBcaptainBbottlesB	apologiesBalanBzealandBzBunitB	recessionB	necessaryBmapBheatherBgwBflickBdaisiesBceremonyBbagelBw00tBtechnicallyBiamjonathancookBforestBfiredBexplodeBerrBdoggieBcurlyBcrackingBchaBbruisedBbonfireBawaitingBattitudeB48BwriterBwhaleBtwistedBpowersBparticularlyBnotebookBniB
mcflyharryBleedsBkkBgooBentirelyBcircleBbobbyBtuBtomoroBsingsBshowersBsalsaBrecommendedBoutlookBmaterialBlambertBhotterBcountsBclickedBboozeBatlanticBahahahaBthedebbyryanBtequilaBshaveB	postponedBmiracleBmcflyforgermanyBfrBfmB	fireworksBeuropeanBcaloriesBbirthBsliceBsilentBsharedBloliBlemonadeBjennyBconversationsBcapeBvillageBserveBmysteryB	lunchtimeBlucyBjackieBhowdBgumBdotBdefaultBcaramelBbrushBboobsB4pmBwifBtmobileBtiedB	spongebobBsonnyB
somethingsBquotitsBloginBjointBihopBheartbrokenBdellBblocksBbelatedBbatBarizonaBamberBuniqueBthesisBsleepinBseptB	salvationBruiningBquarterB
exhaustingBdittoBclimbingBbikiniBnumbBmistakesBmedicalBmaamBloveeeBkfcBholaBhiltonBhaaBessaysBdoseBdisappointmentBdesignerB
depressionBdaneBbuckhollywoodBbitterBayB420ByummBveggieBtornadoBspotsBspelledBsnuggleBskateBrequiredBprocrastinatingBmugBdiggingBcubsBcrawlBaugBasiaB7pmBuhmBsweetestBstuffyBstandardB	scratchedBquotmyBpotatoesBporchBkittiesBhboBdiseaseBdiscountBcouldaBchoicesBcarolinaBboldBursBtampaBsmallerB	realizingBoliveBmodelsBllBjonathanBjessicaveronicaBinformedBflowBdebbiefletcherBchinBbabysitBartsBackBprinterBoiBmaineBhunnyBdonateB	correctlyBwackBsettledBscoredBrobotBrememberingBpadBmaskBloadedBkeenBhomesickBharshBhappilyB	enjoyableBdesignsBcupsBburnsBbbyB800ByaayBturtleB	toothacheBtodayyBsupBriskBproveBkillinBjaredBhuhuBfredBdlBcharmBxoxoxBslapB
productionB	pineappleB	mountainsBlaptopsB	carradineB2008ByellingBsharesBnileyBmisterBmannBleonB	indonesiaBgardensBgalBdiscoB	destroyedBaudienceBanimeB150BwhoopsBtornBthisisdavinaBqueueBqueensBnikkiBnervesBmatthewBhealingBfarewellBdiscussBdaisyBcontemplatingBblockingBannouncementByeahhhBvintageBvacayBunfortunateBstealingBsprainedBminorB
joelmaddenBjeremyBinjuryBfunkyBdryerBdrunkenB	dedicatedBdayyBbomBbasicBwouldaBweedsB	vacationsBstilBromeBoleBnursingBnbcBgpBcumBurselfBtokyoBthirstyB	sunscreenBrickB	returningBpromoteBownedBlotteryBhypeBhmphBextendedBcyclingBcuppaBchargingB	brazilianBbloggerBbiscuitsByelledBwoutBwishinBwinnersBtwittingBtouringBstuffsBretrorewindBresetBpitchBmommysBguestsBgpsBgoodiesBfranBfolkBeatsBdesertBdayquotBcausedBbffsBbballBany1B	addictingB	upsettingBunlikeBstudiedBspoilBslideBseesBraysB
pittsburghB
motorcycleBlistedBlgBhometownBhappendBeyedBermBactressBupgradedBtrialBtodoBstickyBspencerBskippingBpepsiBofficialnjonasBmainlyBkeithBformerBfckBdeletingBcurlBblogtvB	birthdaysBbeboB
unfollowedBtshirtsBstringBscBpalB	oversleptBmacsBloversB
lighteningBjulieBisplayerBfairyBdaddysBbethBantB600B	unpackingBssBorderingBnappingBmodernBmelissaBlightingBlifetimeBgenBfingBchillenB	broadbandBblessingBahhhhhhB70sB1pmB101BtreatingBrussianBraisedBproposalBpokemonBmakerBlionBlectureBkoolBjoesBholdsBgrrrrrBfavesB
consideredBclothingBchargedB	champagneBccBcampaignBanxietyB11thBwreckBwilliamsBtripleB
televisionBstretchBshownBpracticallyBpostersBpirateBphpB
paulaabdulBnowiBmosquitoBlewisBinchBhikeBfeaturedBcoreBcheckinBbrisbaneBbelowBaidB11pmByankeesBwonderedBvictoryBrawrBmotionBlambBhollandBh1n1BcageB
boyfriendsBappropriateBalllB700BwarmingBswapBshtBsackBrockyBregentsBpeasBpaulaBoralBjogBhihiB
governmentBformatBfavsBdirectlyB	coworkersBcasesBasideBandroidBajB55BthickBsunriseBquotifBoldestBmdBexternalBdressingBdatsBcamerasBbcuzBbacheloretteBwireBvacaBtommyBteddyBstadiumBshockingB
sandwichesBrssBrosesBrestoreB
regardlessBoasisBleatherBhealBgahhBelectionBdoughBdescribeBcyaBchristB	botheringB12thB100thBworthyBwhereverBtouchingBtodayiBsuspectBsprayBspammersBsophieBsinkBpuddingBpluginBovaB	ourselvesBlarryBitemBhailBgradesBframeBelsesBdmbBconstantBcondolencesBbrightonBbikingBbgBadminBwrappedBsuitsBshoBshiteBrequestsBrelieveBratB	poisoningBpinBpatioBoatmealBmissyBhorridBgrandmotherBfwdBdraftBcuddlingBcostumeBconfirmationBchilledBcausingBboohooBbettaBaddictBtopsBscaresBmcmahonBjawBintroBgunaBexpertBclaimBblistersBauntieB65BwayyyBwakesBsquirrelB	sleeplessBrootsBrevengeBmeitsBlolsBlesB	lastnightB
jonaskevinBindianaBhoodieBheidiBhandedBelectricityB	educationBboardingBbatmanBtoneBswedishBsnacksBshadesBsendsBringingBrelateB
registeredB	referringBpiercingBowwwB	operationBnetbookBmochaBmikeyBhveBgateBestBdemBcounterBchasingBbettyBbatchBbangsBanthonyB	abandonedB2mozBweedB	unlimitedBtrappedBthtsBtailBperezBmuahBmourningBlimeBlbsBlastsBjoelBindustryBhikingBgawdBformsBduBbrookeBbikesBadobeB30thBtwitterworldBtwitterificB	technicalBreturnsBopensBomgoshBlooongBlearntBjoseBgunsBgalleryBfortuneBfootageBdrugBdirtBdawnrichardBbuffaloBadamsBwtheBwmyBtastedB	switchingBsuckingBsteppedBshucksBsemiBsecretsBscenesBreligionBprogrammingBpowerfulB	portfolioBpoppinBpixarBnoodleBnighterB
highschoolBgcseBcssBcenturyBbruiseB	barcelonaBbahahaBvirginiaBtrainerBstripBstavrosBsrryB	spreadingBshrimpBsewingBselfishBrerunsB
practicingBpourBpairsBneedaBkiddosBhkBfreezeB	financialBfalseBducksBdrakeBdebatingBcraigBbeardB25thBwoooooBwilliamB	volunteerBuiBrodeBrelayB	recognizeBquotwhatBpukeBpoppingBpoleBnoooooooB	neglectedBmwahBmansBlucasBkcBhottestBfollowinBevaBcoworkerBcomicsBcartoonBcaringBwassupBstewartBspidersBpeachBoutdoorBgreaseBdairyB	companiesBboilingBamusedBamongB36BzombiesBvanessaBuuBtunaBsum1BresumeB	obsessionBluggageBistBhowardBembarrassingBelbowBehhBcommingBcarnivalBavoidingBarchieBytBsweatingBquestBpatternBmuchoBfourthBfactorBerrorsBdrumBdenmarkBcompareBcinnamonBchoirBcasaBbushB
birminghamBballetBannualBunderstandingBstickingBsortsBshorterBsentenceB	satisfiedBrightsBkristenstewart9BdrainB	disappearBcrunchBcelticsBcartoonsBrouterBpongBpigsBphotographerBorientationBnonstopBnoisesBnationBlaughterB	jailbreakBit�sBinaperfectworldBheyaBhahhaBgiggleBcultureBcrabBconstructionBcheatingBandyhurleydayBactivityByearbookBspBreadsBorganicBnathanBmollyBmagicalBhatsBfrogBdylanBdodgersB
determinedBdeeBcircusBangelesB1230B11amByellBwolfBtomatoesBteeheeBteasingBstorageBspottedBspokenBsoyBsoakingBrunnyBpumpBparentBneilBmonicaBmeltBlagBhottieBhorriblyBfudgeBffsBfameBepsBegoBbritneyspearsBbeggingBbecomesBballoonBbadassBallllBadvertisingBtrackleBtowerBsymptomsBspecificBsnoringBskatingBquotthisB	promotionB	promisingBpoetryBnowadaysBnevaBmamasBknittingBhomelessB
highlightsBgoalsB	gatheringBfuelBfiguresBficBdunkinBdiskBdancedBbeltB	autographBabsBvitaminBubertwitterBtuesdaysBstrongerBspringsBsheepBrootingBpoundBmultiplyBmowBmethinksBlykBlooooveBlmaooBlighterBjuryBgreatlyBfrankiethesatsBerinBdon´tBcurveBcoldsBbruceB	batteriesBantibioticsBwembleyBwayneB
understoodBtylenolB
respondingB	respondedBrefreshBnestBmurderBmonthlyBimacB	highlightBfilterBdragonBdougBbananasB90210BvisaBviewingBsweetyBsippingBrustyrocketsBrewardBrentsBrelievedBpilotBoffersBloosingBlikeyBjohnsBjerryBjamminBfearnecottonBcrisisB
conventionBchloeBcanonBbegB85BwalksBtastingB	subscribeBshoppinBshadeBrollerBpowBpoisonBopBnascarBloooongBinnocentBholesBhiiBfortBdubaiBdipB	designingBdelishB	confidentBburritoBbrownieBbegunBassumingBapproveBannounceBaliB182ByoungestBthangBsteamBspammingBsaddenedBrolledBracesBpoBpasteBoldsBmickeyBmgiraudofficialBmarsBkennyBimaxBillegalBhundredBhehehBgreeceBdrainedBdefoBbareBagentB	treatmentBtravelsB
travellingBstrepBsolvedBslowerBrove1974BpacificB	messengerBkeBjacksBitiBinfectedBheeheeBfuBflawlessBdraggingBdoucheBcomcastB	broadcastBarghhBapplicationsBakuBwedsBvolumeBunionB
twitterfoxBtreatsBsupplyBsuicideBrantBporBplayoffsBneedlesBlahBjusticeBinfamousBijustineBhersBgloryBcsBcreekBcomplexBbritBbrainsBacctB42BvisionB	treadmillBstevenB	spaghettiBshuffleBsalmonBprovideBmowingBjeBjasonbradburyBindyBidiotsBhubsBenteredBdiscoverBcocoBcnnBchaptersB
attractiveBarrivesBandreaB247BtypesBtemptingBretailBrejectedBnorthernBnatBmonstersBmillionsBleakBkingdomBheidimontagBheartbreakingBhaterBhangoutBearringsBdownsideBbbmBauthorBarrivingBwingB
vegetarianBtwitpicsB
throughoutB	terrifiedB
retweetingBresponsibleB	recoveredBpinkyBpaleBnannyB
membershipBmalaysiaBjuliaBhackBfeeB	exploringBdesignedBcuddlesBcollectB	bowwow614BbootyB	americansBactsB
accountingBwipedB	wallpaperBtranslationBsympathyBsleeepBskippedBsgBscanBremainBovertimeBnmBnatalieBmoonfryeBmeltedBloolBkudosBkongB	irritatedBhaveyoueverBhairsBgulpanagBgotchaB
generationBforeignBemotionsBdiamondB
correctionBconeB	commentedBapprovedBalternativeBalertBagendaB2mrwBzachBupperBtickBsmBsidekickBsharpB	regardingBquothowBprintingBpoopedB
physicallyBmiteBkiddoBjjBjanBhutBhomiesBhighwayBhappiestBgarbageBfurB	dollhouseBdeerB	conditionBclipsBchiliBchaiBburstBblisterBbidBtragedyBsummitBsuiteBsuccessfullyBpaysBleadingBjsBjimmyfallonBhomesBfuzzyBexpoBconfirmB	childrensBcarryingBawwwwwwBannBaloudBallisonBxoxoxoButterlyBuniformBtricksBtnBtivoBtescoBslipB	refreshedBprimeBphraseBpastorBonionBmegBlyBloggingBhungerBhiddenBheaterBfreakyBcooksBcelebritiesB	butterflyBblastingBbenefitBbelgiumBbathingBandorByaaBuveBuhhhBudBtrailBtiresBtattoosBsweaterBspoonBrumBreplayBprogramsBpoundingBoutsBninBmmmmmmB	microwaveBmeltingBkickinBjussBhonoredBfringeBchipotleBbyebyeBbradieBbotB5kBwheelsBwesternBupquotBtreatedBthebrandicyrusBthankingBtemperatureBsyrupB	speciallyBshippedBseekBrussellBruinsBpicksBpathBjournalBgirliesBfreezerBfrankieBfontBfloodedBendlessBegyptBeaBderBclearingBbookingBargBangerB447BwhatchaBsuitcaseBsheeshBpintBpennyBnothingsBlaneBjungleBhayleyBglastoB	developerB
delongedayB	cellphoneBcarrieBcallinBboBargumentBannoyBabB33BwhomBtraceB	suggestedBsidesBrubbingBpsychBpaypalB	naturallyBmisssBmentBmaiBkrispyBfilipinoBfightsBdreamtBclaireBcheatB	championsBattachedBantonioByayyyyBvomitB
unemployedBrsBpandoraBnvmBnateBmillionaireBmemphisBmedBhoeBcopeBcoloursBcasinoBbusesBauBalikeBadaB14thB10kBtypedBtwentyBtravisB	surprisesBstiffBsteelBsolveBshoreBshakesBseeinBrealllyBpresenceBnytBmatchesB
masterchefBmachinesBlotsaBlollBjunBhittinBguessedBgrinBfeedsBdavidsBcrcBcordBcommuteBcoconutBbuffetBbondBashtonBahahahBwardrobeBtwittsBtinaBtimelineBsissyBshaunjumpnowBsarcasmBrefusesBprintedBpillBoregonBoffendedBisraelBhumansB	hollyoaksB	groceriesBflippinBfattyB
engagementBdennysBcuB
confidenceBcobraBclinicBcheekBcalB3000BzoeBtriesB
surroundedBsupriseBsumthinBspeakerBsimpsonsBscottishBsafetyBpingBpaceBmmitchelldavissBlistingBlinkedBkarenBickBhubbysBgrannyBgooseBfrickinBdwBclubsBbamB29thB26thB–ByurByungBvariousBunbelievableBtreyB	transportBtiBskinsBsims3BservingBresortB	remainingBplotBpleaseeeBoriginBniecesBmadisonBlatinBjewelryBjaBinnerBgrantBgbBexpiredBdavisBcraftBcodyBbounceB
appearanceBannieBaaBwntBtpBthailandBtcotBsueBslumdogBsheetBsampleBrestlessBrashB	purchasedBpiratesBphaseBperoBnicolerichieBnadaBmileymondayBklBjanuaryBheeyBh8BgueBgmaBghostbustersBferBdecB	countriesBcommunicationBbyeeBbrutalBbrickBbraB24thBtyrese4realBtwittBtowelBtestedBswagBsacBroomieBpeakB
passengersBowwBmadridBleoBlatersB
experimentBexitedBdumpedB	downloadsBcodesBchrisdjmoylesBcharlesBbuffyB	blueberryBapplyingB22ndByaaaayBxmasBwompBtwatBtutorialBtedBstickamBsandalsBrestedBrangersBramenBrallyB	newspaperBmustveBmonkeysB	milkshakeBledBkindleBjadeB	interfaceB	hurricaneBfordBestateB
disturbingBdecidingB
connectingBceilingBbubblesBbrowsingBamandapalmerBwalkinBuppBstayinBshldBseoBprotectB
pretendingBoutingBmasBivyBitchingBinvitesBictBhumBhammerBevertonBentertainedBdwighthowardBdrsB
classmatesBcanalBbelfastB	banksyartBanneBydayBwearsBtrvsbrkrBtowersBtoriBtidyingBsystemsBstrokeBrickyBrecipesBpondBmoiBmarieBleopardBknowwBillnessBiconsBgoddamnBfellaBeyebrowsBeasternBdestroyBdarlinBdanielleBcrackersBcourtneyBcookoutBclearedBchicaBcamperBboostBbastardsBaightB
accidentlyBabbyBzomgBunknownBtwistBtomsBtoddBstandsBspyBsmashBslippedBromanceBrentalBquotweBofcourseBnoisyBnephewsBmapsBleaderBknifeBhenryBgleeBfluffyBdebutBdebateBcozyBcestBcaseyBbrittanyBblushBbenjaminBantsB900BwiB	universalB
timefollowBsurvivorBstinkBsnakeBrepBquotinB	publishedBontdBmrtweetBmaybB
margaritasBloanB	laughhaveBkansasBinventedBhardwareBforthBflavorBdisagreeBbannerBalaskaB64B2007BsmellyBshannonBpaydayBoxfordBmudBltltBlistsB
invitationBintendedBhunterBgoodbyesBgiB
foundationBflopsBexplainsBdocumentaryB
departmentBdavidhenrieBcullenBbucketBbrighterBbackedBamountsB
activitiesBwpBswallowB	strangersBsoundingB
preorderedBpersonalityBpanelBnuBnicestBmorganBmaniBlosesBlackingBjenniferBhurrahBfrozeBcopiesB	baltimoreB27thB
withdrawalBwhiskeyBwendysBwaaaayBtablesBsuppliesBspreeBspectacularBshadowBpiercedBninaBmumbaiBmanualBlittlefletcherBlabelBkissedBkidneyBkaylaBjohnsonB
impressionBflagBfianceBexitBespressoB
disappointBdewBcocktailBciderB
aubreyodayBwiffB
watermelonBtomoBseminarBronBrcBpunBpeelingBmiloBmeeeeeBleahBleadsBirlBikrBhamsterBgranBgiantsBgalsBfkBfinlandBearnedBdumpBdraggedBdisconnectedBconverseBcodBanaByukBtbsBtaraBsinBservedBredoBpolicyBpaneraBooooohBnicksantinoB	nickjonasBngaBnetballBnearbyBmethodBmentallyBlcBlautnerBimoBgrammarBftlB	discoveryBcrowdedBchaosB	backwardsBautomaticallyBarrestedBadmireB23rdBymBweddingsBversionsBveggiesBveB	underwearBtriviaB
trampolineBtnxB
thoroughlyBtherBsuckerB
soooooooooBrottenBroadsBpsychedBoooooBmovinBkewlBjtBfittingB	exclusiveBeraBdodgyB
directionsB
devastatedB
chocolatesBbudsBashesBalyssaBaleBabuseB250BzackBwethetravisBvisualBunclesBtinkB	thousandsBthisisrobthomasB	receivingBquotdontB
psychologyBperuBoscarBnavyBliBlappyBl8rBfocusedBfloodBdirectorBdestinationBdancerBcolumbusB	cocktailsBbooooooBbaliByouiB	translateBthnksBstruggleBspeedingBsmackBshavedBshampooBpremierBorleansBomBnauseousBleakedBjammingBiowaB	invisibleBinlawsBincreaseBincludesBincidentBgilmoreBdinerB
colleaguesBclassyB	christinaBchopBbumpedBboardsBalphaBadditionB630amBtabBstareBseparateBsausagesB	regularlyBpunchedBportugalBpfftBnerveBnanaBmediumBlotionBlaxBiranianBinputBinchesBhopefulBghettoBgeeksBgahhhBenviousBdyedBdiningB	dependingBdealsBcrawlingBcheatedBceptBbeccaBathensBasthmaBwaaayBtollBtissuesBspoilersBsooonBrockstarB
rehearsalsBpcdBparkedBoverwhelmedB	organizedBmisBmeowBmamB	magazinesBlawsBlaborBjulianBjessemccartneyBhtmlBhomeeB	eachotherBduperBdosentBdisabledBdhughesyBdesireBcardiffBcandlesBbroadwayBbitingBbillionBbangingBbanBauctionBacademyB110ByummmBwarmerB
thomasfissBtasksBstormingBsozBsnowingBshuttleB	pricelessB
philosophyB
originallyBnovBnonsenseBmjBmetsBlaurensBichBhiringBgoooBfreaksBfolderBeditedBdelhiBcooperBcitiesBcapitalB	breakdownBboreddBbordersBwivBthsBsubmitBstonesBstitchesBsqueezeBsammyBpodcastsBpaymentBpacketBnowwBliquorBhassleBgrabbedBgoooooodBfenceBdoughnutB	corporateBconcernB	cigaretteBchatsBbrillBbreastBboxingBbondingBautumnBarghhhBwheredBwendyBunluckyBtypoBtwittedBsupermanBstereoBscaringB	requestedBregistrationBrangBrageBquotohBpuertoB	permanentB	patientlyBoutdoorsBmodemBmixtapeBlungsBlt3333BlousyBlimitsBkiddiesBinsanelyBicingBhushBhusbandsBchosenBbristolBbittersweetBartworkBaliensB2hrsBzuneBvisitorsBvegB	upgradingBtrackingBtomarrowBt20BstinkyBstickerBsowwyBranchBplanesBpancakeB
organizingBomjBoffenseBnicknameBneedleBitzBeeekBdormBditchedBcostcoBcockB	assistantBwhipBtmwBstickersB	sensitiveBsearchedBscrollBrichmondBrestartB	relativesB
regrettingB
protectionBplatformBpauloBokayyB	nominatedB
newsletterBlovequotB	introduceBhotelsBgordonBgadgetBfiestaBelephantBdintBddayBdatabaseBcrownBcouncilBcolinBciaoBcardioBbleachBbenefitsBavB	auditionsB1capplegateBvoicesBufcBtroublesBtayBsoakBscoresBsansBquotgoodBpleaseeBpersonsBpaycheckBnobodysBmsgsBmajorlyB
irritatingBiloveyouBhurrayBheeeyBgutsBderekBdamagedB
conclusionBchadBcausesBbryanBamazinB41BwuzBwelBvisitsBviolenceB	veronicasBthemesBteenagerBteenageBstartinBsneezeBsallyBrocketBpollBpgBoklahomaBnikeB	neighbourBmastersBmannnBmajorityBlivinBladygagaBjennettemccurdyBinspirationalBhowdyBhamiltonBguardBgotoBexperiencedBespnBdwightBdrivinBdriedBdecidesBdasBcontBcomplimentsBbacB72BtuitionBrunninBretreatBrequiresBrbBprimaryBpreorderBpedicureBpauseBooopsBnoahBnetherlandsBlbBjuicyBicarlyBgrapesBempireB	electionsBdivingB
discussingBdeprivedBcreditsB
creativityBcracksBclickingBbunBalgebraB97B530amB4gotBygB	wrestlingBwhackB	twittererBtutBtreasureB	temporaryBtagsB	submittedBspiceBsnifflesBshitsBrisingB	reinstallBredbullBpeekBoddsBnonethelessBmixingBmitchellB	migrainesBmealsBmbBmariahBjuzBiplayerB	impatientBglueBfundsBflippingBfierceBfieldsBexpectationsBemployeeBeddieBearlBdayiBbecuzBbeckyBbaileyBbackingBawaitsBaidenBwomensBtipsyBtcBroadtripBringtoneBprixBnicksBnachosBmovementBmosBmeanieBlindaBkmBjbsB	influenceBimprovedBhumpBhopelessBhashtagBgovtBgoogledBglobeBgigglesBfateBenemyBencouragementB	elizabethBdongBdiaryBdexterBcoveringBcostaB
commentaryBclosesBcleanerBbfsBbblBasdaBahemBxddBtweetinBsmellingBsamsBrumorBohhhhhBnatalBmidtermBknowwwBjennBjdBindexB	homeworksBhobbyBhmmmmmBharpersBgutBgoodsexBgloomBfacialBexperiencingBenteringBembarrassedBelvisBdrummerBdistrictBdentalBdaniBcubeBconnectionsBbeatenBalthoB	yorkshireBwheatBwayyBwatchesButahBtjBtherealtiffanyBtaxesBslaveB	selectionBseafoodBraininBquickerBpythonBpeedBouBnuthinBneilhimselfBmensBmaggieBlesbianBjobrosBjakartaBindieBheheheheBenjoyinBengagedB	elsewhereBeditorBdreamedBdearlyBcourtesyBconvertBcatchyBbutterfliesBarenaBangelaBalllllBalienBagencyBнеBzzzBtehranBtaiBslimBricoBrealhughjackmanBratsB	quothappyB
portugueseBpokeBpodBodBmileysBlaunchedBkungBjpBjayzBinspireBgotaBgomezBgokeyBgitBforeheadBforcingBflatleyBeuroBenvironmentB	economicsBconfuseBcharmingBcassieBbriefBbridalB	bookstoreBboiBbanksBadvancedB98BxxxxxxBwifesBunwellBtinBtilaBtigersBtabletsBstrikesBstreakBstanleyBsecretlyB	screeningBsaoBrogersBpropertyB
permissionBnvrBnemoBmoodsBmeetsBleftoverBkissingBinfrontBidealBhellsBheatingBguiltBgroveBformalBfirmwareBfictionBfallonBdistractingBcheeredBchecksBbckB96B38BwelliBvetsBunusualBtourneyBtooooooBterminalBshontellelayneBsettlingBscreamedBsailingBrentedBreachingBpcsBmbpBkremeBiraqBhundredsBhomepageBhindiBgrandfatherBgapBfortunatelyB
developingBdeeplyBcottonBcolorblindfishBchewBcedarBcarrotBcabinBbasketB	backstageBargueBarBadultsBadoptBwreckedBwhiningBverBtiffanyBtiffBthnkBtheirsB	tennesseeBsurveyB
straightenBspiritsBsockBsneakyBskittlesB	showeringBsethBpumpkinBpineBpalsBnewbieBn97BmurrayBmodBlonesomeBkaren230683B	graduatesBfacingB
delightfulBcondoBconditioningBcmtBclapBcheekyBbullsBbrewBbleedBbaldB	badmintonBbacksBawwhBarrivalBarntBanyonesBactorsB28thB1111ByestBwweBvibeB	tetheringBstringsBstrandedBsavannahBrumorsB	responsesBpromisesBpollenBpinsBpenisBpatronBnanBmellowBkindlyBkathyBjtimberlakeBjosephBjohnlloydtaylorBinvitingBferryBentriesBeaseBdiddyBcooolBbedsBbeachesB	animationBaidanBadvertB2getherBиBwldBwhyyyBweeeBwarrantyBvivaBventBtsBtimequotBterrificBstungBseedsBsadfaceBponyBpiesBoverloadB
optimisticBoliverB	nostalgicB	nooooooooBlindsayBlasagnaBlancearmstrongBjlsofficialBjillBimyBhtcBhongBhighestBgravityBgalaxyBfobBfabricBdozenBdoomBdeniedB
definitionBconnorBchampionBaaaahB69B39B10000B	wikipediaBwanB
underneathBuhohBtabsB	supportedBstormyBsolangeknowlesBsmokedBshelterBsellsBscifiBrunnerBretireBrelaxinBregardsBpbBnotificationsB
neglectingBnancyBjeanBheycassadeeBglowBgeoBfondBdoggiesBdmvBddubBcottageBcoffeBchillsBboreBadelaideB46B44BwalleBunfortunatlyBtwhirlBtoddlerBswellBsoggyBscarfBscaleBrubberBretweetsBresearchingBrecallBproteinB	politicalBparaBpaigeBownersB	overratedBoutcomeBojB	manhattanBlisaveronicaBlineupBlalaBjudgingBjaysB	hahahahahBglovesBgaspB	freelanceBfistBeepBdeffoBcomebackBcoleBcoincidenceB	classroomBchartBbuckBbruisesBborderBbeganBawfullyBadoptedByangBworkersB
wonderlandBwillieday26BtweeplesBtutorBthirtyBstudiosBshelfBseesmicBrushingBrobbedBrllyB
revolutionBquotallBqldBproduceBoutletBoddlyBnathanfillionBmussoBmommasBmentionsBlooveBlendBkentuckyBkardashiansBinnBhowreBhoooBhonorsocietyBfreshmanBfloatingBfestivitiesBdon�tB	developedBdeafBdangitBcrispBcrackerBcowsBcolumnBclayBbanquetBanyhowBajaB	acceptingB56BwesBugghBtossBtofuBticklemejoeyB	throbbingBtellinBtackleBstruckBspitBrdB	raspberryBpussyBpossibilityBpimpBoreosBontarioBomggBofferingBnewlyBmrazBmelodyBmeantimeBmarcBliverB	lifestyleBlandingBkeynoteBgearsBgabeBfuriousBfogBfighterB
feliciadayBfedexBfactsBexcusesBethanBellieBdreadedBdivaBdingBdeffBdbBcouponBbotsBblahhBbadgeBairplaneBwhydBthrowsB	stephanieBsocalBseshB	sarcasticBsaneBrelationshipsBregBraidBquittingBpuffyB	promotingBproductivityBpjBownsBnerdsB	margaritaBmarcusBkeeperBiniBgiddyBfinnaBdrearyBdegrassiBdearestBdashBcan´tBbeaBattacksBalrightyBaffairB100000BвByippeeByeaaBwoundBwelllBviolinBunlovedBtropicalBslamBslackingBshawnBshavingBsamanthaBrequireBrepublicB
reasonableB	practicalBpilatesBoyBnowwwBnearestBncisBnaoBmauBjambaBironicBimprovementB	greggarboBgrantedBgolfingBfoolsB	equipmentBdrowningB	doughnutsBdancesBcluelessBcaveB43ByknowByeaahBwoeBvampiresBtempleBsupportsBshuttingBsharonBromeoB
relaxationBquotsoBquotheyBpupsBprofBouttBmaidBlikBlickBlengthBkrisallenmusicBinternalBincBfunnnBfireflyBdtBdistractionBdfizzyBdevelopBdetailBdefenseBdebtBcriminalBcoollikeBcastingBbuhBbnBblurryBbahamasBarguingB120BwateringBurgentBupiBundBunbelievablyBtrentBtemporarilyBsubscriptionBstevieBstatBsplashBspillBsourcesBsnoozeBsniffleBsipBsexualBscooterBsagaBrustyBrebootBquizzesB	programmeBprocrastinationBplazaB	pembsdaveBpasBpartayBomgggBmikesB	meanwhileBmarvinBloveeeeBlikewiseBliarB	leftoversB	instantlyBhuhuhuBhahahahahahaBgeekyBfunkBfroBfeesB	extensionBexploreB
exhibitionBewwwwBdiggBdebbieBdancersBcowboyBcoloredBcarlosBbellsBalabamaBafricanB62ByesssssBvbsBtradingBtorrentBswitzerlandBstationsBspeltBschooolBsayangBroundsB	reportingBperfumeB
mentioningBmayerBlouBliquidBkristinBkasiB	heartburnB
gailporterBfinBenuffBenufB	effectiveBdwnBdrivenBditchBdarkerBcnBcanucksB	cambridgeBbrunoBallowsBagonyBadvilB102B08B¬¬BwitchBweaknessBwavingBuniteBtunnelBtruBtrippinB	thursdaysBstrategyBsnappedBsilverstoneBrepostBpeachesBparticipateBoverseasBoliviaB	officallyBmuhBmonopolyBmarkingBmangaBmampmsBlogicBjelousBislandsBiamthecommodoreBhoppingBhollieBgreedyBfinishesBfcB	exceptionB	earphonesBdkBdianaBcurledBcheeseburgerBbunniesBblakeBbicycleBassesBassemblyBapproachBalancarrB2bB2006B1500BzackalltimelowByeayByarnBwittyBwiredBvouBvocalBtdBsubjectsBsosBsharkBsbBrushedB	replacingBpottyBpointingBpenguinBoverdueBottawaBofficesBmkB	miserablyBmeetupBmaddieBlolaBknowiBi’mBimaginationBhurtinB	householdBhookahBhiiiBhashBfinanceBexplanationBdistractBdehBcocoaBchilliBbrightenBboltBbarryBanalysisBamericasBafterallBaddsB��BwindsBwhyyBustreamBunrealBtributeBtannedBshoutingBshelbyBsamplesB	reviewingBresidentBrehabBrapeBquotjustBpoloBplacedB	performedBnoeBmerchBmarylandBloudlyBlegoBketchupBjamieoliverBitchBinvolveBgtgBfowardBfishyBfifthBexperiencesB
elementaryBdescriptionBchuckmemondaysBchairsBballoonsBbagelsBavailBattBahhaB5000BwerBvocalsBvacuumBturkishBtrapBtrainedB
supposedlyBsherrieshepherdBscreensBriBradarBproblyBprintsBpetitionBpainkillersBoreoBnativeBmoodyBmnBmargaretBliamBlegallyB	languagesBkenBhawtBginoandfranBexplorerBeffortsBdramaticBdigginBdevonB
developersBdepotBdemisB
cigarettesBchoosingBcheeksBcategoryBcameronBbuggyBbelleB	attackingB
assessmentBappetiteBanywhoB88B430amBwhoooBwatersBwalaBvelvetBtrimBtommBtherealsavannahBteaserB	strangelyBsodBshinesBremainsBrecBquotwhyBplantingBperspectiveBnkBnicholasBmiiBlenosBlaurieBlabourBiplBinstructionsBincomeBikBhumourBhoedownBhahahhaBgirlquotBformulaBfeatBfeastB
explainingBduetBdockBdelaysB	deadlinesBcrampBcindyBcherriesBcentsBbodiesBbloggersBbiscuitBbingoBattendedBallyBahhhhhhhBacidBaccBabroadB37BzipByeahiBwonderfullyB	wisconsinBwilsonBweightsBvictimsBvictimBughhhhhBthnBterryBtakBsyndromeB
surprisingB
statisticsBspockBsistaBsandyBsadderB
riandawsonBrearBraveBquotgetBpreparationBportableBokieBnoticingBmiseryBminimumB
medicationBmannyBjarBinternBintelligentBinjuriesBhatz94BgetawayBgavinBequallyBefronBeeeBdiveBcriticalBcpBcourageB
compatibleBcomedyqueenBcarrotsBcarlB	bluetoothBbarbieB51BveniceBtrippedBtodayyyB
threadlessB	splittingBslutBsanaB	remembersBremedyBremakeBpresentationsBpreppingBparksBorganizeBoctBnicBnahhBmadreBlosersBlincolnBlatterBkeepinBjerksBjeepB	jayme1988BjackedBittBironyBhumbleBhipsB	hangoversBgoodsBglastonburyBghostsBftskBfallinBdrillBcoursesBcoopBcommunicateBciaraBcaraBbfastB	bangaloreBbackkBancientBaltonBaloeBallsBabsentB58B4wardByewBxboxe3BwarnedB	wanderingBvinylBtrickyBtoreBspeedyBsooooooooooBsleeeepBshhhBsaddBrockbandB
recommendsBpublishBpluggedBpencilBpediBoutquotBnudgeBmarginatasnailyBkettleBheathBhardyBhackingBhabitsBfoughtBfleaBequalBehhhBdryingBdrupalBcontextBchilisBceBbehaveBarsenalBalleyB78B311BwellsBventureBtruelyBtlcBtitlesBsymphnysldrBswissBsubscribersB	sprinklesB	sheffieldBprofilesBpillowsBpamBpackagesBnikonBnerdyBmatureBmampgBkyBjerrysB
introducedBheyyyyBhddBgrowsBfsBfiB
everybodysBequalsB	employeesB
dictionaryBdangerBdammBcuterBconfBclownBclimateBclarksonB	christineBchickensBboaBbewareBamongstBagreesBachyB140confB1200BwivesBwidgetB	voicemailBunavailableBtranceBtierdBtherellBstableBslBskillBsizesBsequelBrosieBrolandBriotBrenewedBreliableBquotnotBmyyBmockBmermaidBmatBmamiBmafiaBlucascruikshankBlmaoooBlensesBlaserBknitBkeriBirvineBincaseBin2BhubBhatinBgovBfrickenBfountainB
favouritesBfalloutBevansBepiphanygirlB	designersB
controllerBchampBcarolBbuggingBbubbaBbluntBbendBbenchBavenueB105BwipeBvalidBunlikelyBunfollowingBthyBtherealshaqBteethingBtearingB
speechlessBsomthingBskintBsailorBrefusedBrebelBratedB	prototypeB
processingBprizesBprincesssupercBpresaleB	placementBokiBmoniesBkiwiBingredientsBimpactBhypedBhhrsBheavilyBharBgriefBgreetingBgraveBfebBeventfulBemailingBdsiBdrawnBdownerBdimBdavejmatthewsBdanecookBcryinB
continuingBbutiBbulletB	bobbyllewBbloggedBbizarreBbarnBawhhBauntyB
atmosphereBappreciationBaffectedB125ByeapBwaxBvicBuptoBultraBtokioBstewartkrisBspeaksBsparksBsomBsitsBsickyBshootsBrussiaBrelyBrachBquotloveBpplsB
playgroundBphoBpeepBoriginsBmuBmooreB	metallicaBmercyBmaddBloooooveBlettuceBkanyeBjsutBharrisBgravyBfundB
friendsterBfloatBflatsBfinalyBfascinatingBdrivewayBdivineBdayyyBcpuBconsoleB	confusionBcoloringBclothdiapersBcharBbuBbtsBbradleyB
bradiewebbBblairBbailedBattemptsB4everBwhalesBtmBtksB	thepistolBsweepBsizedBrnBrestaurantsBrebeccaBquotyourBpowderBposhBpolandBplantedB	pinkberryB	paparazziBouchieBnovaBniggasBmommiesBmikeywayBlvBkayaBjumperBjulietBjcBiiiB	halloweenBhaftaBgreaterBginBfussBfrownBexcelBeveningsBericaBencouragingBdrownBdreadBdowntimeBdontyouhateBdeptBdaylightBcoralineBconditionerB
commentingBchoppedB	chillaxinBblurayBblipfmBbittenB	believingBbakeryBanoopdoggdesaiBalohaBzeByouuuuuByayaBwormBwoohBwizardsBwayyyyBvodafoneBveronicaButterBusageBunpackBtwitterrificBtechnoBstephanieprattBsouljaBslidesBsamsungBsambergBroomiesBrobotsBreserveBpuzzleBphishBoutfitsBopinionsBnurseryB
noooooooooBmojoBmimiB	milwaukeeBmiddayBleaningBleanBhorizonBhelmetBheightsBhammockBgoatBgentleBfoggyBfilmedBfartBfaintBeughBdialB
craigslistBcoachingBchristopherBchargesBblurBassholesBaquariumBandersonBadrianB	absolutlyB82ByanBwagonBvarietyBtitsBthisisryanrossBthighsBtf2BtearyBt4BsubsBstrollB	standardsBslotBscrabbleBsavesB	ringtonesBretroBrenewBpukingBprovidedBpajamasBp90xBoutageBoohhBonionsBneonBmildBmeuBmelbBmcdsBmatthewsBmailsBladBjqueryBjordinBinvolvesBhaaaBguitarsBgivinBgenerousBgemB
expressionBdpBdesBdaynightBdaycareBdanaBcrunchykBcouplesBclubbingBbmwBbeautifullyBaudreyBareasBaidsByaaaBwin7BwhyyyyBwazBvh1BurlsBtyposB	throwdownBtabletBstoopidBshiaBshapedBredsBrackB
punishmentBpoohBordinaryBnewportBmunichBmultiBmmwantedBmilitaryBmakeoverBlinedBleakingB
leadershipBkirkBjoshtastic1B	honeymoonBhairdresserBguyzBgrooveBgrillingBgooooBgoldfishBfuckenB
friendfeedBfreshlyBflyladyBfloodingBdatelineBcrystalchappellBconvosB
celebratedBbritainBbestfriendsBbattlefieldBbarbequeBarsedBapprovalB	applebeesBangieBaffectB52B07BynBwitnessBwelshBviceButBunsureBunderstandsBummmmBspecificallyBscoutB	satelliteBsashaBsamantharonsonBsaddensBretiredBrerunBrereadB	quotyoureBpugBps2B	professorBpreviewsBpocketsBpeppersBpatternsBoverlyBopportunitiesBnauseaBmushroomBmtgBmountBmilanBmarketsBlouiseBlipstickBkickassBjennaBituBinterviewedBindoorBhugglesBhuggedBhomeeeBgreensBgrapeBflakesBfiledBfckingBfamilysBdumbassBcrunchyB
cinderellaBcarbsBbehalfBbarrelBasylmBarrangeBanthemByasminaBwrappingB	worldquotBworkiBwikiBvlogBtwiiterB
timberlakeB
thunderingBsupermarketBsuperbBspecsBsleeveBsavingsBruBquotwhenBquotitBpriorBpremiumBposeBperformancesBothB
oliviamunnBnutellaBnasaBmisskeribabyBmirandaB
mcflymusicBmatchingBlonerBkindnessBjackmanBisleBhttpwww4officeautomationcomB	hoppusdayBhopinBhelplessBhelenBfuzzballBfavourBfadingBemailunlimitedBdroolBdownhillBdoooBdomesticBcreepB
crackberryBcmBchampsBcamdenBcalgaryBbranchBblink182BbahahahaBashleyltmsyfBannoysBweighB	unhealthyBtireddB
temptationBspanBsolarB
satisfyingB
robluketicBrinBretardBrestoredBreadersBrandyB
quothiquotBpriorityBpatientsBoccasionBnswBnobleBnawwBmauiBmailingBliteBlapsBkeza34BjanetBiyaBindependentBhqBhattonBguineaBgrandadBgownBglBfloorsBfletcherBexistsBevidenceBevanB
electronicBcueBcookinBcolaBchesterBchapBcartBbyeeeB
bronchitisBbrewingBbrettBbarkingB68B49B3oh3B103ByeeBwoopsBviBudahBtwelveBtopshopBteensBsyncingBsunbatheBsiblingsBshaheenBscreamsBquotandBpixieBphilliesBparBosxBnihBnhlBnetworksBmutualBmultitaskingBmistakenBloriBladsBinsertBichatB	hospitalsBhavenBgrrrrrrBgabrielsaportaB
extensionsBelliotBeagleBdreadfulBcrispsB	crazinessB	copyrightB
confessionBcombinedBcollarBcirclesBcan�tBcansBbuatBborrowedBbecuaseB3hrsBwasherBwarcraftBvietnamBunlockBtouchesBtomorroBstellaB	sociologyBsmoresBsmokinB
shortstackBshinBrepairsBreferBracistBquotnewBproducerBpiBpawBmusicianBmotorBmoduleBmartiniBmajornelsonBlobbyBlalalaBintroducingBhaleyBhairyBgtaBgrabbingBgeometryB	genuinelyBfrisbeeBferrariB	featuringBemotionallyBdistantBdiceB	delightedBconcreteB
complaintsBchokedBblahhhBbeleiveBbeijingBbasisBbangkokBaussiesBauchB	attemptedBapproachingBalexisBalbertBairlinesBadjustB1sB
whatsoeverBwakinBvickyBunproductiveB	tutorialsBtooiBtbBtallerBsponsorBspiltB	snugglingBsinusesB	scrambledBscarB	roommatesBreducedBrankBrainbowsB
perfectionBoughtBoracleBnickyBmunchingBmuchlyBmuchhBmeimBluxuryBlungBloyalBlolololBliftingBlagiBkartB
journalismBjollyBitttBindiansBholBheaderBgucciBgrahamBgelBexplodedBerasedBellaBdrenchedBdonnaBdojieBdeniserichardsBdeltaB
dedicationBdahBcurlsBcrackinB
commissionB
chesterdayBcalebBblokeBblackoutBbentB
banksyart2BaustriaBarkansasB
alyankovicBaddthisB54B47B12amBнаByeahhhhBwilwBtresBtowardBtowBtisdaleBthankiesBtalagaBstudyinB	stockholmBstartupBspinachBspencerprattBslicedBshytBseeeBsealBreverseB	releasingBraBpstBpoutBpouredBpointedBpoemsBpipeBpiggyBpermitBoutstandingBnyquilBmunchiesBmooBmarinaBmansionBlimoBkevBkellyrowlandBjasmineBjamsB	improvingBhiredBhangsBfuckcityBfosterBforkBfoldingBflowingBfellasBelevenBdissertationBdirectedBdinnersBdiabetesBdenyBdentistsBdefeatBcurlingB	committedBclumsyBcloneBcjBchuffedBcalvinBburiedBbritsBbbsBbb10BappearedBanhBaaaB92ByipeeBwohooBwhitB	welcomingBwakeyBwaaahBu2BtrophyBtracyBtotesBtonfueBstaceyBsleeeeepB
situationsBsickkBshanghaiBrpBrocketsBreadinBratesBraisingB
powerpointBpossibilitiesBpickyBpalaceBpacksBnowquotBmonsoonBloganBlifequotBleafBlampBkanBjelloBjealousyBipodsBharveyBguildBgraciasBgeneBg2gBfresnoBfirmBdocumentBdjingBdiyBdepositB	crucifireBcremeBcoreyB	coachellaBclassicsBchocoBbuenosBbroadcastingBbrandyBadapterBaboardBzoomBwritesBwhootBwhippedBweirdestBwakeupBwaistBtwitterbugsBtraditionalBthrBswellingBstarshipBsnakesBsketchBsitterBshowcaseBshoutsBshadyBscrewingB
scratchingBroastedBquotdoBprotestB	pregnancyBnhBmichaelsBloungingBlockerBjlsBjennifalconerBiquotmBibizaBhidBgrubBgoofyBgladlyBfrustrationBfreewayBflirtingBdukeBdinosaurBdakotaBcudntBchunkBchokeBchicBchanelBcellsBcarterBcandleBbuysBbouncingBboredddBbackpackBanticipationBanimatedBachievementBaccurateB106ByankeeBx3BwoohoooBtxtingBtwiggasBtrucksBtrayBtotBtodaybutBthierBteesBtaleBsweeetBstupidlyBsplendidBsmoothlyBremovingBreleasesBregretsBreduceBrecievedBpickleBosloBoakBmobBmiceB	messagingBmcrBlolllB
literatureBliftedBknockingBipBgumsBgrBgettnBfunnierBfruitsBfoulBfoldBeu09BdualBdistractionsBdeliBdebitBdaytimeBdannysBdancinBcvBcruzBcornyB	continuedB
comparisonB
chroniclesBcasualBbrushingBbraceBblondB	blokeslibBbarbecueBbaddB	awesomelyBarcadeB
activationB
accomplishB115ByooBxxxxxxxBwsBwoofBwbBwahhhBvalBunlockedB	unchartedBtissueBtingBthankssBstinkinBsthBsteadyBshaqBsecsBrippingBreminiscingBregionBquotoneB
preferablyBpoolsideBpolarBpmsBperiodsB
passionateB	nessie111B	minnesotaBmattressB	marvelousBmammaBluvsBloveliesBlottaBlibBjudezxoBjasperBizzyB
inevitableBheelBhampmBfunctioningBfuckersBembraceBeddieizzardBdivorcedBdeyBdestinyBcupboardB	consumingBchuckleBchrisdaughtryBchitownBchasedBcapableBbumperBbrittBblanketsBbiggieBawwwwwwwBapaBamoBakBairlineB730amB57BworkerBwithdrawalsBunoBunemploymentBughiBtoastedBthesims3BtempsBsuchaBsuBsteppingBspikeBspacesBsoulsBsnuggledBslumberBsinglesBsermonBserialBselectedBsedBsarahsBrevBreportedBquidBprettierBprankBpnkBoppsBomggggBokeyBnotificationBnitesBmowerBmishaBmattsBmanicBlinkedinBlimpingBleaseBjustifyBintegrationBillinoisBhaulBharleyBharborBgriffinBglitterBginaBfifteenBfadedBerrrBegBdugBdosBdonkeyBdisapointedBcwBcrispyBcreedBcrankBcommandBcologneBclearsBchewingBchessBcheeriosBcharlieskiesBcanyonBbrianmcnuggetBbebeBbbqsBarvoBanoopBanoBallllllB67BywBwizardBwillyBwhaBvehicleBundergroundBtsarnickBtrendsBtoursB
substituteBsquashBsoothingBsoldierBsingersBshizB
sacramentoBropeB	resourcesB	representB
relativelyBrejectsBrefundBpissesBpierBpatrolBoutaBnowbutBnedaBmustardBmuggyBmotivateB
mosquitoesBmmvasBmattyB
lobosworthBlawyerBjustinmgastonBitbutBgradersBgoddessBgainingBfilthyBengBebookBdjsBdefeatedBdebB	creaturesBcrammingB
conditionsBchartsBchallengingBcarolineBcapacityBcaitlinBbravoBboobBbollocksBallowingB
allnighterB	againquotBadmittedBéBzenBunderstandableBuggBtrousersBtakersBswiftsBswBsposeBsoarBsixthBsicklyBselectBryansBragingBpullsBpropBpressedBplatesBpeelBpalaBnooooooooooBnayB
mysteriousBmpBmissouriBmgmtBmayoBlindseyBintendB	insomniacBinformBimhoBie6BhiatusBheavensBgoodieBghBgarrosBfuseBfnBflopBexpensesB
exercisingBeasiestB
decoratingBdecadeBcvsBcommBcharmsBcanvasBbrasilBbetsBbeginingB	bandwidthBanklesBamazinggB	accidentsB94B66B2gB1100ByeaaahB	worldwideBworkkBwoowBwindingBwigBwhineBverwonBtysonBturtlesBtrBthotBtheatersBtemplateB	superstarBsummeryBstingBsqlBslackBscoopB	schedulesBpsychoBproviderBpoolsBpimpleB
officialasBoccasionallyB	obnoxiousB
nottinghamBmorrowBmitBmilBmanlyBlpBlitterBlataBlargerBkiteBjessieBjensenBjasonmanfordBjaiBitchesBinsBinitialB
individualBhiphopBhahahaaBfossiloflifeBforcesBfiresBfearlessBfaireBeurghBemotionBelementBdroolingBdetoxBcontainB
collectingBchuBchorusBchippedBcertificateBcarriedBbrothaBblockbusterBbeverlyBatchaB	alexanderB	alcoholicBaddyBactionsBaccentsB93B1hrBwhitneyBwarnBvolunteeringBveraBuncoolBtymBtweopleB
terrifyingBstomachacheBsmiledBsmashingB	signatureBrodBroastingBriskybusinessmbBrhymesBqualifyBpshB	protectedBpreventB	ponderingBphxBnightiBmysqlBmurderedB	mosquitosBmcraddictalBlangfordperryB
kevinjonasB	jasonmrazBirBinvestBimmenseBhuBhttptwitpiccom6q1omBhttptrimlvbuBhttptinyurlcomry9wapBhoesBhillsongBhawkcamBgooglingBgenuineBfreebiesBfilingBfebruaryBfadeBduvetBdunoBdireBdiplomaBdifficultiesBdawBconsolationBcomaBcareyBcaredBcaptureBbumsBanythinB911B60sB59B330amB02ByeshBwherBwaitsBwahooBvickytcobraBverseBtraceyhewinsBtotalyB	thenewbnbBtanksBtangoBstubbornBstripedBstrawBseedBscrapedB
rescheduleB	religiousBrapedBrafaelBquotquotBpussycatBpinchBowenBnyaBmishacollinsBmillerBleslieBlandlordBkuyaBkentBkelseyBjaydenicoleBinsultB	ignoranceBhuggingBhaikuBgroovyBgamerBescapedBerghBdqBdevoBcurtainsBcs4BcraveBcountedBcontrolsB	colleagueBclanBcharliesBbunchesBbonnarooBassumedBagainiBabsenceB830amByeppBworkoutsBwbuBwardBwalkerBvettelBvaginaBteleBteheBsuprisedB
subscribedBspoilerBspammerBsophiaBslammedBsemB	quotthatsB
prioritiesB
presentingBpoliteBobvBobrienBnickybyrneofficBmothBmillB	mccartneyBmaccasBloveyouBlogiesBlillyBlabsBkarateBinsistBheavenlyBharmBgreasyBfooBfagBentranceBenjoysBdslBdomB	diagnosedBcreationBcornwallB	catherineBbonjourBbelongsBbargainBbabygirlBaskinBappointmentsBantiBamazeBalexsBachieveB30minsBzonesByayyyyyByasBwormsBwmeBwhysBvmBtonsilsBtodayandBteemwilliamsBswingsBswampedB
supportiveBstomacheB	spidermanBsobsBsneakingBshowtimeBscriptsBruthBrefusingBrecommendingBprivacyBpredictBpppBphilsBpendingBpawsBowieBmodelingBmoanBmereBmapleBmananaBlooolBlanBkerryBjunoB
investmentBinterventionBintBinstallationBimportantlyB	impalaguyBimmuneBiaBhostedBhondaBheadlineBhawaiianBgearingB	evolutionB
esmeeworldB
eliminatedBelevatorBdjalfyBdelightBdampBcuntBcommitBcolumbiaBcolderBchriscornellBchowB	chickfilaBcheckoutB
charlestonBcelebrationsBbrBboatsBbitchyBbarnesBarthurB	anxiouslyBairingBaawB50thB104BwtBwooowBwheeBw8BviolentBviennaBulcerBtweetheartsB	traditionBtossingBtobyBtlkBthatquotBtaylorsB	sweetnessBstuffingBstirBstaplesBstalkersBspecBsneakersBsinginBsharksBsethsimondsBschedBsaintsBrihannaB	retweetedB	rereadingBredwingsBrascalBplayboyB
pakcricketBonequotBmoneysBmitchBmidwestBmicroBmarBlumpBleightonBlanceBlagunaBl4dBkiB	jimmycarrB
javascriptB	inventoryBimaginedBiamspectacularB
hystericalBhsmBhomoB	heartbeatBharlemBhanksBhandlingBgummyBgrimB	greatnessBgpaBgatesBgaryveeBfootieBextendB
exhaustionBdrewryanscottBdoomedBdonatedB	dividendsBdiscoveringBdisappearingBdevilsBdependBdaftB
crazytwismB	corruptedB	correctedB	convertedBcolorfulBcoasterBchattedBcarlisleBcanberraBbothersBboobooBblushingBbloatedBbecBbarbaraB
babysitterB	automaticBairedB4getB31stByanksBwolvesBwelcomedBwatchnBwarmthBvonBvanishedBtoasterBtapingBswordB	swallowedBsufferedBsquadBsoooooooooooBsoooonBsnickersBslashBservesBschBroxyBrollinBrelativeBreinstallingB	preschoolB	pleaseeeeBpfBpeanutsBpartnersBpandamayhemBpakistanBotwBorchidflowerBneeedBmuyBministryBmarshallBmanicureBmalBlooooongBlkBlicenceB	launchingB	landscapeBkennedyBkalebnationBjumpsBjgBissBhealsBhartluckBhahaiB	guaranteeBflavourBeligibleBeagerBdustyB	dreamlandBdonationBdominosBdoeBdarknessBcyberB
contagiousBcliffBcircumstancesBchoBcatholicBcabinetBbuzzingBbrandsBbordBboatingBblizzardBbelievedB	amazingggBadviseB4rmB1000thBzeldaB
yourselvesBwooohoooB
unbearableB	teenagersBstrictBsneezedBsixteenBsidewalkBshareholderBscrapBscottyBscamBresponsibilityB
resolutionBrenoBrelevantBrazorBpuppetBpuffBphotobucketBphantomBpattyB
parliamentBpantiesBouchiesB	orchestraB	norwegianBmussomitchelB	musiciansBmunchBmphBmoralBmerlinBmentorBmeinBmbaBlovBlottoBkiddinBjtvBjoblessB
jaylastarrBitschelseastaubB
inspectionB	injectionBiloveBidentityBheightB	hairsprayBgsBgraderBgagBfucksBfuckerBflyerBexpenseBeeBdxB	disturbedBdieingBcursedBbullyBbrakeBborinBbdB	appealingB
apartmentsBaliciaBalexandramusicBairconB
additionalB77BzzzzByoghurtB
wwwm2easiaBwhoohooB	whispererBwarriorBwaaaaayBvergeBupsetsBtvsBthasB
tessmorrisBtalesBsydBsvuBstlBsklBsippinB
sharepointBscrubBriderBrealsBrailsBrabbitsBquottoBquotgtBquakeB
positivityBportraitBplayoffBpickupB
photographBperksBpartysBowwwwBofficialchariceBmistyBmigraneBmaisBllsBliningBlibertyBksBjustineB	interwebsBinterruptedB
intentionsBiluBhideousBhawksBgcsesBfxBfluidBflorenceB	encourageBdoritosBdineBdilemmaBdifferentlyBdifferencesBcreepingB	cranberryBcoryBcollageBclashBchubbyBchirpingBcharsBcacheBbreakupBbeddBanberlinBaimingBagedB
acceptableB89B79B71ByawnsBxtraBxsBwahhBtyraBtwiterBtimeeBtiaBthreatBtamBtaggedBstudBstinkingBstampBshabbyBsconesBsaunaB	sandwhichBrytBrpattzBrobsBrevisedBrescued
??
Const_5Const*
_output_shapes	
:?N*
dtype0	*??
value??B??	?N"??                                                        	       
                                                                                                                                                                  !       "       #       $       %       &       '       (       )       *       +       ,       -       .       /       0       1       2       3       4       5       6       7       8       9       :       ;       <       =       >       ?       @       A       B       C       D       E       F       G       H       I       J       K       L       M       N       O       P       Q       R       S       T       U       V       W       X       Y       Z       [       \       ]       ^       _       `       a       b       c       d       e       f       g       h       i       j       k       l       m       n       o       p       q       r       s       t       u       v       w       x       y       z       {       |       }       ~              ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?                                                              	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?       	      	      	      	      	      	      	      	      	      		      
	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	       	      !	      "	      #	      $	      %	      &	      '	      (	      )	      *	      +	      ,	      -	      .	      /	      0	      1	      2	      3	      4	      5	      6	      7	      8	      9	      :	      ;	      <	      =	      >	      ?	      @	      A	      B	      C	      D	      E	      F	      G	      H	      I	      J	      K	      L	      M	      N	      O	      P	      Q	      R	      S	      T	      U	      V	      W	      X	      Y	      Z	      [	      \	      ]	      ^	      _	      `	      a	      b	      c	      d	      e	      f	      g	      h	      i	      j	      k	      l	      m	      n	      o	      p	      q	      r	      s	      t	      u	      v	      w	      x	      y	      z	      {	      |	      }	      ~	      	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	       
      
      
      
      
      
      
      
      
      	
      

      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
       
      !
      "
      #
      $
      %
      &
      '
      (
      )
      *
      +
      ,
      -
      .
      /
      0
      1
      2
      3
      4
      5
      6
      7
      8
      9
      :
      ;
      <
      =
      >
      ?
      @
      A
      B
      C
      D
      E
      F
      G
      H
      I
      J
      K
      L
      M
      N
      O
      P
      Q
      R
      S
      T
      U
      V
      W
      X
      Y
      Z
      [
      \
      ]
      ^
      _
      `
      a
      b
      c
      d
      e
      f
      g
      h
      i
      j
      k
      l
      m
      n
      o
      p
      q
      r
      s
      t
      u
      v
      w
      x
      y
      z
      {
      |
      }
      ~
      
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                                      	       
                                                                                                                                                                  !       "       #       $       %       &       '       (       )       *       +       ,       -       .       /       0       1       2       3       4       5       6       7       8       9       :       ;       <       =       >       ?       @       A       B       C       D       E       F       G       H       I       J       K       L       M       N       O       P       Q       R       S       T       U       V       W       X       Y       Z       [       \       ]       ^       _       `       a       b       c       d       e       f       g       h       i       j       k       l       m       n       o       p       q       r       s       t       u       v       w       x       y       z       {       |       }       ~              ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?        !      !      !      !      !      !      !      !      !      	!      
!      !      !      !      !      !      !      !      !      !      !      !      !      !      !      !      !      !      !      !      !      !       !      !!      "!      #!      $!      %!      &!      '!      (!      )!      *!      +!      ,!      -!      .!      /!      0!      1!      2!      3!      4!      5!      6!      7!      8!      9!      :!      ;!      <!      =!      >!      ?!      @!      A!      B!      C!      D!      E!      F!      G!      H!      I!      J!      K!      L!      M!      N!      O!      P!      Q!      R!      S!      T!      U!      V!      W!      X!      Y!      Z!      [!      \!      ]!      ^!      _!      `!      a!      b!      c!      d!      e!      f!      g!      h!      i!      j!      k!      l!      m!      n!      o!      p!      q!      r!      s!      t!      u!      v!      w!      x!      y!      z!      {!      |!      }!      ~!      !      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!       "      "      "      "      "      "      "      "      "      	"      
"      "      "      "      "      "      "      "      "      "      "      "      "      "      "      "      "      "      "      "      "      "       "      !"      ""      #"      $"      %"      &"      '"      ("      )"      *"      +"      ,"      -"      ."      /"      0"      1"      2"      3"      4"      5"      6"      7"      8"      9"      :"      ;"      <"      ="      >"      ?"      @"      A"      B"      C"      D"      E"      F"      G"      H"      I"      J"      K"      L"      M"      N"      O"      P"      Q"      R"      S"      T"      U"      V"      W"      X"      Y"      Z"      ["      \"      ]"      ^"      _"      `"      a"      b"      c"      d"      e"      f"      g"      h"      i"      j"      k"      l"      m"      n"      o"      p"      q"      r"      s"      t"      u"      v"      w"      x"      y"      z"      {"      |"      }"      ~"      "      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"       #      #      #      #      #      #      #      #      #      	#      
#      #      #      #      #      #      #      #      #      #      #      #      #      #      #      #      #      #      #      #      #      #       #      !#      "#      ##      $#      %#      &#      '#      (#      )#      *#      +#      ,#      -#      .#      /#      0#      1#      2#      3#      4#      5#      6#      7#      8#      9#      :#      ;#      <#      =#      >#      ?#      @#      A#      B#      C#      D#      E#      F#      G#      H#      I#      J#      K#      L#      M#      N#      O#      P#      Q#      R#      S#      T#      U#      V#      W#      X#      Y#      Z#      [#      \#      ]#      ^#      _#      `#      a#      b#      c#      d#      e#      f#      g#      h#      i#      j#      k#      l#      m#      n#      o#      p#      q#      r#      s#      t#      u#      v#      w#      x#      y#      z#      {#      |#      }#      ~#      #      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#       $      $      $      $      $      $      $      $      $      	$      
$      $      $      $      $      $      $      $      $      $      $      $      $      $      $      $      $      $      $      $      $      $       $      !$      "$      #$      $$      %$      &$      '$      ($      )$      *$      +$      ,$      -$      .$      /$      0$      1$      2$      3$      4$      5$      6$      7$      8$      9$      :$      ;$      <$      =$      >$      ?$      @$      A$      B$      C$      D$      E$      F$      G$      H$      I$      J$      K$      L$      M$      N$      O$      P$      Q$      R$      S$      T$      U$      V$      W$      X$      Y$      Z$      [$      \$      ]$      ^$      _$      `$      a$      b$      c$      d$      e$      f$      g$      h$      i$      j$      k$      l$      m$      n$      o$      p$      q$      r$      s$      t$      u$      v$      w$      x$      y$      z$      {$      |$      }$      ~$      $      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$       %      %      %      %      %      %      %      %      %      	%      
%      %      %      %      %      %      %      %      %      %      %      %      %      %      %      %      %      %      %      %      %      %       %      !%      "%      #%      $%      %%      &%      '%      (%      )%      *%      +%      ,%      -%      .%      /%      0%      1%      2%      3%      4%      5%      6%      7%      8%      9%      :%      ;%      <%      =%      >%      ?%      @%      A%      B%      C%      D%      E%      F%      G%      H%      I%      J%      K%      L%      M%      N%      O%      P%      Q%      R%      S%      T%      U%      V%      W%      X%      Y%      Z%      [%      \%      ]%      ^%      _%      `%      a%      b%      c%      d%      e%      f%      g%      h%      i%      j%      k%      l%      m%      n%      o%      p%      q%      r%      s%      t%      u%      v%      w%      x%      y%      z%      {%      |%      }%      ~%      %      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%       &      &      &      &      &      &      &      &      &      	&      
&      &      &      &      &      &      &      &      &      &      &      &      &      &      &      &      &      &      &      &      &      &       &      !&      "&      #&      $&      %&      &&      '&      (&      )&      *&      +&      ,&      -&      .&      /&      0&      1&      2&      3&      4&      5&      6&      7&      8&      9&      :&      ;&      <&      =&      >&      ?&      @&      A&      B&      C&      D&      E&      F&      G&      H&      I&      J&      K&      L&      M&      N&      O&      P&      Q&      R&      S&      T&      U&      V&      W&      X&      Y&      Z&      [&      \&      ]&      ^&      _&      `&      a&      b&      c&      d&      e&      f&      g&      h&      i&      j&      k&      l&      m&      n&      o&      p&      q&      r&      s&      t&      u&      v&      w&      x&      y&      z&      {&      |&      }&      ~&      &      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&       '      '      '      '      '      '      '      '      '      	'      
'      '      '      '      '      '      
?
StatefulPartitionedCallStatefulPartitionedCall
hash_tableConst_4Const_5*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *%
f R
__inference_<lambda>_2466437
?
PartitionedCallPartitionedCall*	
Tin
 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *%
f R
__inference_<lambda>_2466442
8
NoOpNoOp^PartitionedCall^StatefulPartitionedCall
?
?MutableHashTable_lookup_table_export_values/LookupTableExportV2LookupTableExportV2MutableHashTable*
Tkeys0*
Tvalues0	*#
_class
loc:@MutableHashTable*
_output_shapes

::
?P
Const_6Const"/device:CPU:0*
_output_shapes
: *
dtype0*?P
value?PB?P B?P
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
	variables
	trainable_variables

regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
;
	keras_api
_lookup_layer
_adapt_function*
?
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
?
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses
"_random_generator* 
?
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses

)kernel
*bias*
?
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/__call__
*0&call_and_return_all_conditional_losses
1_random_generator* 
?
2	variables
3trainable_variables
4regularization_losses
5	keras_api
6__call__
*7&call_and_return_all_conditional_losses

8kernel
9bias*
?
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses

@kernel
Abias*
<
1
2
)3
*4
85
96
@7
A8*
<
0
1
)2
*3
84
95
@6
A7*
* 
?
Bnon_trainable_variables

Clayers
Dmetrics
Elayer_regularization_losses
Flayer_metrics
	variables
	trainable_variables

regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
Gtrace_0
Htrace_1
Itrace_2
Jtrace_3* 
6
Ktrace_0
Ltrace_1
Mtrace_2
Ntrace_3* 
* 
?
Oiter

Pbeta_1

Qbeta_2
	Rdecay
Slearning_ratem?m?)m?*m?8m?9m?@m?Am?v?v?)v?*v?8v?9v?@v?Av?*

Tserving_default* 
* 
7
U	keras_api
Vlookup_table
Wtoken_counts*

Xtrace_0* 

0
1*

0
1*
* 
?
Ynon_trainable_variables

Zlayers
[metrics
\layer_regularization_losses
]layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

^trace_0* 

_trace_0* 
_Y
VARIABLE_VALUEdense_24/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_24/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
?
`non_trainable_variables

alayers
bmetrics
clayer_regularization_losses
dlayer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses* 

etrace_0
ftrace_1* 

gtrace_0
htrace_1* 
* 

)0
*1*

)0
*1*
* 
?
inon_trainable_variables

jlayers
kmetrics
llayer_regularization_losses
mlayer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses*

ntrace_0* 

otrace_0* 
_Y
VARIABLE_VALUEdense_25/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_25/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
?
pnon_trainable_variables

qlayers
rmetrics
slayer_regularization_losses
tlayer_metrics
+	variables
,trainable_variables
-regularization_losses
/__call__
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses* 

utrace_0
vtrace_1* 

wtrace_0
xtrace_1* 
* 

80
91*

80
91*
* 
?
ynon_trainable_variables

zlayers
{metrics
|layer_regularization_losses
}layer_metrics
2	variables
3trainable_variables
4regularization_losses
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses*

~trace_0* 

trace_0* 
_Y
VARIABLE_VALUEdense_26/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_26/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

@0
A1*

@0
A1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
_Y
VARIABLE_VALUEdense_27/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_27/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
5
0
1
2
3
4
5
6*
$
?0
?1
?2
?3*
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
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
V
?_initializer
?_create_resource
?_initialize
?_destroy_resource* 
?
?_create_resource
?_initialize
?_destroy_resource><layer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/*
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
<
?	variables
?	keras_api

?total

?count*
M
?	variables
?	keras_api

?total

?count
?
_fn_kwargs*
z
?	variables
?	keras_api
?true_positives
?true_negatives
?false_positives
?false_negatives*
z
?	variables
?	keras_api
?true_positives
?true_negatives
?false_positives
?false_negatives*
* 

?trace_0* 

?trace_0* 

?trace_0* 

?trace_0* 

?trace_0* 

?trace_0* 

?0
?1*

?	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

?0
?1*

?	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
$
?0
?1
?2
?3*

?	variables*
ga
VARIABLE_VALUEtrue_positives_1=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEtrue_negatives_1=keras_api/metrics/2/true_negatives/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEfalse_positives_1>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEfalse_negatives_1>keras_api/metrics/2/false_negatives/.ATTRIBUTES/VARIABLE_VALUE*
$
?0
?1
?2
?3*

?	variables*
e_
VARIABLE_VALUEtrue_positives=keras_api/metrics/3/true_positives/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEtrue_negatives=keras_api/metrics/3/true_negatives/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEfalse_positives>keras_api/metrics/3/false_positives/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEfalse_negatives>keras_api/metrics/3/false_negatives/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
?|
VARIABLE_VALUEAdam/dense_24/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_24/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?|
VARIABLE_VALUEAdam/dense_25/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_25/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?|
VARIABLE_VALUEAdam/dense_26/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_26/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?|
VARIABLE_VALUEAdam/dense_27/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_27/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?|
VARIABLE_VALUEAdam/dense_24/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_24/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?|
VARIABLE_VALUEAdam/dense_25/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_25/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?|
VARIABLE_VALUEAdam/dense_26/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_26/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?|
VARIABLE_VALUEAdam/dense_27/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_27/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
serving_default_input_4Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCall_1StatefulPartitionedCallserving_default_input_4
hash_tableConstConst_1Const_2dense_24/kerneldense_24/biasdense_25/kerneldense_25/biasdense_26/kerneldense_26/biasdense_27/kerneldense_27/bias*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *.
f)R'
%__inference_signature_wrapper_2465923
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename#dense_24/kernel/Read/ReadVariableOp!dense_24/bias/Read/ReadVariableOp#dense_25/kernel/Read/ReadVariableOp!dense_25/bias/Read/ReadVariableOp#dense_26/kernel/Read/ReadVariableOp!dense_26/bias/Read/ReadVariableOp#dense_27/kernel/Read/ReadVariableOp!dense_27/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp?MutableHashTable_lookup_table_export_values/LookupTableExportV2AMutableHashTable_lookup_table_export_values/LookupTableExportV2:1total_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp$true_positives_1/Read/ReadVariableOp$true_negatives_1/Read/ReadVariableOp%false_positives_1/Read/ReadVariableOp%false_negatives_1/Read/ReadVariableOp"true_positives/Read/ReadVariableOp"true_negatives/Read/ReadVariableOp#false_positives/Read/ReadVariableOp#false_negatives/Read/ReadVariableOp*Adam/dense_24/kernel/m/Read/ReadVariableOp(Adam/dense_24/bias/m/Read/ReadVariableOp*Adam/dense_25/kernel/m/Read/ReadVariableOp(Adam/dense_25/bias/m/Read/ReadVariableOp*Adam/dense_26/kernel/m/Read/ReadVariableOp(Adam/dense_26/bias/m/Read/ReadVariableOp*Adam/dense_27/kernel/m/Read/ReadVariableOp(Adam/dense_27/bias/m/Read/ReadVariableOp*Adam/dense_24/kernel/v/Read/ReadVariableOp(Adam/dense_24/bias/v/Read/ReadVariableOp*Adam/dense_25/kernel/v/Read/ReadVariableOp(Adam/dense_25/bias/v/Read/ReadVariableOp*Adam/dense_26/kernel/v/Read/ReadVariableOp(Adam/dense_26/bias/v/Read/ReadVariableOp*Adam/dense_27/kernel/v/Read/ReadVariableOp(Adam/dense_27/bias/v/Read/ReadVariableOpConst_6*8
Tin1
/2-		*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference__traced_save_2466602
?
StatefulPartitionedCall_3StatefulPartitionedCallsaver_filenamedense_24/kerneldense_24/biasdense_25/kerneldense_25/biasdense_26/kerneldense_26/biasdense_27/kerneldense_27/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateMutableHashTabletotal_1count_1totalcounttrue_positives_1true_negatives_1false_positives_1false_negatives_1true_positivestrue_negativesfalse_positivesfalse_negativesAdam/dense_24/kernel/mAdam/dense_24/bias/mAdam/dense_25/kernel/mAdam/dense_25/bias/mAdam/dense_26/kernel/mAdam/dense_26/bias/mAdam/dense_27/kernel/mAdam/dense_27/bias/mAdam/dense_24/kernel/vAdam/dense_24/bias/vAdam/dense_25/kernel/vAdam/dense_25/bias/vAdam/dense_26/kernel/vAdam/dense_26/bias/vAdam/dense_27/kernel/vAdam/dense_27/bias/v*6
Tin/
-2+*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *,
f'R%
#__inference__traced_restore_2466738??
?
.
__inference__destroyer_2466387
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?

?
E__inference_dense_26_layer_call_and_return_conditional_losses_2465398

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:

identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????
a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
E__inference_dense_25_layer_call_and_return_conditional_losses_2465374

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
E__inference_dense_24_layer_call_and_return_conditional_losses_2466255

inputs1
matmul_readvariableop_resource:	?N-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?N*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????N: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????N
 
_user_specified_nameinputs
?
?
/__inference_ffnn_on_count_layer_call_fn_2465449
input_4
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:	?N
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:

	unknown_8:

	unknown_9:


unknown_10:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_ffnn_on_count_layer_call_and_return_conditional_losses_2465422o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_4:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
<
__inference__creator_2466374
identity??
hash_tableo

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name	2041754*
value_dtype0	W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: S
NoOpNoOp^hash_table*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
ݨ
?
#__inference__traced_restore_2466738
file_prefix3
 assignvariableop_dense_24_kernel:	?N.
 assignvariableop_1_dense_24_bias:4
"assignvariableop_2_dense_25_kernel:.
 assignvariableop_3_dense_25_bias:4
"assignvariableop_4_dense_26_kernel:
.
 assignvariableop_5_dense_26_bias:
4
"assignvariableop_6_dense_27_kernel:
.
 assignvariableop_7_dense_27_bias:&
assignvariableop_8_adam_iter:	 (
assignvariableop_9_adam_beta_1: )
assignvariableop_10_adam_beta_2: (
assignvariableop_11_adam_decay: 0
&assignvariableop_12_adam_learning_rate: M
Cmutablehashtable_table_restore_lookuptableimportv2_mutablehashtable: %
assignvariableop_13_total_1: %
assignvariableop_14_count_1: #
assignvariableop_15_total: #
assignvariableop_16_count: 3
$assignvariableop_17_true_positives_1:	?3
$assignvariableop_18_true_negatives_1:	?4
%assignvariableop_19_false_positives_1:	?4
%assignvariableop_20_false_negatives_1:	?1
"assignvariableop_21_true_positives:	?1
"assignvariableop_22_true_negatives:	?2
#assignvariableop_23_false_positives:	?2
#assignvariableop_24_false_negatives:	?=
*assignvariableop_25_adam_dense_24_kernel_m:	?N6
(assignvariableop_26_adam_dense_24_bias_m:<
*assignvariableop_27_adam_dense_25_kernel_m:6
(assignvariableop_28_adam_dense_25_bias_m:<
*assignvariableop_29_adam_dense_26_kernel_m:
6
(assignvariableop_30_adam_dense_26_bias_m:
<
*assignvariableop_31_adam_dense_27_kernel_m:
6
(assignvariableop_32_adam_dense_27_bias_m:=
*assignvariableop_33_adam_dense_24_kernel_v:	?N6
(assignvariableop_34_adam_dense_24_bias_v:<
*assignvariableop_35_adam_dense_25_kernel_v:6
(assignvariableop_36_adam_dense_25_bias_v:<
*assignvariableop_37_adam_dense_26_kernel_v:
6
(assignvariableop_38_adam_dense_26_bias_v:
<
*assignvariableop_39_adam_dense_27_kernel_v:
6
(assignvariableop_40_adam_dense_27_bias_v:
identity_42??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?2MutableHashTable_table_restore/LookupTableImportV2?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:,*
dtype0*?
value?B?,B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEBFlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table-keysBHlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table-valuesB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/3/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/3/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/3/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/3/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:,*
dtype0*k
valuebB`,B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::*:
dtypes0
.2,		[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOp assignvariableop_dense_24_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_24_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_25_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_25_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_26_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_26_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp"assignvariableop_6_dense_27_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp assignvariableop_7_dense_27_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_iterIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_beta_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_beta_2Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_decayIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOp&assignvariableop_12_adam_learning_rateIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0?
2MutableHashTable_table_restore/LookupTableImportV2LookupTableImportV2Cmutablehashtable_table_restore_lookuptableimportv2_mutablehashtableRestoreV2:tensors:13RestoreV2:tensors:14*	
Tin0*

Tout0	*#
_class
loc:@MutableHashTable*
_output_shapes
 _
Identity_13IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOpassignvariableop_13_total_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOpassignvariableop_14_count_1Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOpassignvariableop_15_totalIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOpassignvariableop_16_countIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp$assignvariableop_17_true_positives_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOp$assignvariableop_18_true_negatives_1Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp%assignvariableop_19_false_positives_1Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp%assignvariableop_20_false_negatives_1Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp"assignvariableop_21_true_positivesIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp"assignvariableop_22_true_negativesIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp#assignvariableop_23_false_positivesIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOp#assignvariableop_24_false_negativesIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_dense_24_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_dense_24_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOp*assignvariableop_27_adam_dense_25_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOp(assignvariableop_28_adam_dense_25_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOp*assignvariableop_29_adam_dense_26_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOp(assignvariableop_30_adam_dense_26_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_31AssignVariableOp*assignvariableop_31_adam_dense_27_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_32AssignVariableOp(assignvariableop_32_adam_dense_27_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_33AssignVariableOp*assignvariableop_33_adam_dense_24_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_34AssignVariableOp(assignvariableop_34_adam_dense_24_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_35AssignVariableOp*assignvariableop_35_adam_dense_25_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_36AssignVariableOp(assignvariableop_36_adam_dense_25_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_37AssignVariableOp*assignvariableop_37_adam_dense_26_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_38AssignVariableOp(assignvariableop_38_adam_dense_26_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_39AssignVariableOp*assignvariableop_39_adam_dense_27_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_40AssignVariableOp(assignvariableop_40_adam_dense_27_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_41Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_93^MutableHashTable_table_restore/LookupTableImportV2^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_42IdentityIdentity_41:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_93^MutableHashTable_table_restore/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "#
identity_42Identity_42:output:0*i
_input_shapesX
V: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
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
AssignVariableOp_2AssignVariableOp_22*
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
AssignVariableOp_3AssignVariableOp_32*
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
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92h
2MutableHashTable_table_restore/LookupTableImportV22MutableHashTable_table_restore/LookupTableImportV2:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:)%
#
_class
loc:@MutableHashTable
?C
?
__inference_adapt_step_2465971
iterator9
5none_lookup_table_find_lookuptablefindv2_table_handle:
6none_lookup_table_find_lookuptablefindv2_default_value	??IteratorGetNext?(None_lookup_table_find/LookupTableFindV2?,None_lookup_table_insert/LookupTableInsertV2?
IteratorGetNextIteratorGetNextiterator*
_class
loc:@iterator*
_output_shapes
:	?*
output_shapes
:	?*
output_types
2Y
StringLowerStringLowerIteratorGetNext:components:0*
_output_shapes
:	??
StaticRegexReplaceStaticRegexReplaceStringLower:output:0*
_output_shapes
:	?*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite u
SqueezeSqueezeStaticRegexReplace:output:0*
T0*
_output_shapes	
:?*
squeeze_dims

?????????R
StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
StringSplit/StringSplitV2StringSplitV2Squeeze:output:0StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:p
StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        r
!StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       r
!StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
StringSplit/strided_sliceStridedSlice#StringSplit/StringSplitV2:indices:0(StringSplit/strided_slice/stack:output:0*StringSplit/strided_slice/stack_1:output:0*StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_maskk
!StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
StringSplit/strided_slice_1StridedSlice!StringSplit/StringSplitV2:shape:0*StringSplit/strided_slice_1/stack:output:0,StringSplit/strided_slice_1/stack_1:output:0,StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
BStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast"StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
DStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast$StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
LStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeFStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
LStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
KStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdUStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0UStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
PStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreaterTStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0YStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
KStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastRStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxFStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0WStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
LStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2SStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0UStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulOStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximumHStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimumHStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
OStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountFStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0WStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
IStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
DStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumVStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
MStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
IStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
DStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2VStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
UniqueWithCountsUniqueWithCounts"StringSplit/StringSplitV2:values:0*
T0*A
_output_shapes/
-:?????????:?????????:?????????*
out_idx0	?
(None_lookup_table_find/LookupTableFindV2LookupTableFindV25none_lookup_table_find_lookuptablefindv2_table_handleUniqueWithCounts:y:06none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
:|
addAddV2UniqueWithCounts:count:01None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:?
,None_lookup_table_insert/LookupTableInsertV2LookupTableInsertV25none_lookup_table_find_lookuptablefindv2_table_handleUniqueWithCounts:y:0add:z:0)^None_lookup_table_find/LookupTableFindV2",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
 *(
_construction_contextkEagerRuntime*
_input_shapes
: : : 2"
IteratorGetNextIteratorGetNext2T
(None_lookup_table_find/LookupTableFindV2(None_lookup_table_find/LookupTableFindV22\
,None_lookup_table_insert/LookupTableInsertV2,None_lookup_table_insert/LookupTableInsertV2:( $
"
_user_specified_name
iterator:

_output_shapes
: 
?
e
G__inference_dropout_10_layer_call_and_return_conditional_losses_2466270

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
*__inference_dense_27_layer_call_fn_2466358

inputs
unknown:

	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_27_layer_call_and_return_conditional_losses_2465415o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????
: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
??
?
J__inference_ffnn_on_count_layer_call_and_return_conditional_losses_2465654

inputsS
Otext_vectorization_4_string_lookup_9_none_lookup_lookuptablefindv2_table_handleT
Ptext_vectorization_4_string_lookup_9_none_lookup_lookuptablefindv2_default_value	0
,text_vectorization_4_string_lookup_9_equal_y3
/text_vectorization_4_string_lookup_9_selectv2_t	#
dense_24_2465631:	?N
dense_24_2465633:"
dense_25_2465637:
dense_25_2465639:"
dense_26_2465643:

dense_26_2465645:
"
dense_27_2465648:

dense_27_2465650:
identity?? dense_24/StatefulPartitionedCall? dense_25/StatefulPartitionedCall? dense_26/StatefulPartitionedCall? dense_27/StatefulPartitionedCall?"dropout_10/StatefulPartitionedCall?"dropout_11/StatefulPartitionedCall?Btext_vectorization_4/string_lookup_9/None_Lookup/LookupTableFindV2`
 text_vectorization_4/StringLowerStringLowerinputs*'
_output_shapes
:??????????
'text_vectorization_4/StaticRegexReplaceStaticRegexReplace)text_vectorization_4/StringLower:output:0*'
_output_shapes
:?????????*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite ?
text_vectorization_4/SqueezeSqueeze0text_vectorization_4/StaticRegexReplace:output:0*
T0*#
_output_shapes
:?????????*
squeeze_dims

?????????g
&text_vectorization_4/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
.text_vectorization_4/StringSplit/StringSplitV2StringSplitV2%text_vectorization_4/Squeeze:output:0/text_vectorization_4/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
4text_vectorization_4/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
6text_vectorization_4/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
6text_vectorization_4/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
.text_vectorization_4/StringSplit/strided_sliceStridedSlice8text_vectorization_4/StringSplit/StringSplitV2:indices:0=text_vectorization_4/StringSplit/strided_slice/stack:output:0?text_vectorization_4/StringSplit/strided_slice/stack_1:output:0?text_vectorization_4/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask?
6text_vectorization_4/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
8text_vectorization_4/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
8text_vectorization_4/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
0text_vectorization_4/StringSplit/strided_slice_1StridedSlice6text_vectorization_4/StringSplit/StringSplitV2:shape:0?text_vectorization_4/StringSplit/strided_slice_1/stack:output:0Atext_vectorization_4/StringSplit/strided_slice_1/stack_1:output:0Atext_vectorization_4/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Wtext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast7text_vectorization_4/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
Ytext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast9text_vectorization_4/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
atext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShape[text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
atext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
`text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdjtext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0jtext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
etext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
ctext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreateritext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ntext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
`text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastgtext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
ctext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
_text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMax[text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0ltext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
atext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
_text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2htext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0jtext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
_text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMuldtext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0ctext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum]text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0ctext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum]text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0gtext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
dtext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincount[text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0gtext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0ltext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
^text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ytext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumktext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0gtext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
btext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
^text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ytext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2ktext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0_text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0gtext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
Btext_vectorization_4/string_lookup_9/None_Lookup/LookupTableFindV2LookupTableFindV2Otext_vectorization_4_string_lookup_9_none_lookup_lookuptablefindv2_table_handle7text_vectorization_4/StringSplit/StringSplitV2:values:0Ptext_vectorization_4_string_lookup_9_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
*text_vectorization_4/string_lookup_9/EqualEqual7text_vectorization_4/StringSplit/StringSplitV2:values:0,text_vectorization_4_string_lookup_9_equal_y*
T0*#
_output_shapes
:??????????
-text_vectorization_4/string_lookup_9/SelectV2SelectV2.text_vectorization_4/string_lookup_9/Equal:z:0/text_vectorization_4_string_lookup_9_selectv2_tKtext_vectorization_4/string_lookup_9/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
-text_vectorization_4/string_lookup_9/IdentityIdentity6text_vectorization_4/string_lookup_9/SelectV2:output:0*
T0	*#
_output_shapes
:??????????
3text_vectorization_4/string_lookup_9/bincount/ShapeShape6text_vectorization_4/string_lookup_9/Identity:output:0*
T0	*
_output_shapes
:}
3text_vectorization_4/string_lookup_9/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
2text_vectorization_4/string_lookup_9/bincount/ProdProd<text_vectorization_4/string_lookup_9/bincount/Shape:output:0<text_vectorization_4/string_lookup_9/bincount/Const:output:0*
T0*
_output_shapes
: y
7text_vectorization_4/string_lookup_9/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
5text_vectorization_4/string_lookup_9/bincount/GreaterGreater;text_vectorization_4/string_lookup_9/bincount/Prod:output:0@text_vectorization_4/string_lookup_9/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
2text_vectorization_4/string_lookup_9/bincount/CastCast9text_vectorization_4/string_lookup_9/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 
5text_vectorization_4/string_lookup_9/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
=text_vectorization_4/string_lookup_9/bincount/RaggedReduceMaxMax6text_vectorization_4/string_lookup_9/Identity:output:0>text_vectorization_4/string_lookup_9/bincount/Const_1:output:0*
T0	*
_output_shapes
: u
3text_vectorization_4/string_lookup_9/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R?
1text_vectorization_4/string_lookup_9/bincount/addAddV2Ftext_vectorization_4/string_lookup_9/bincount/RaggedReduceMax:output:0<text_vectorization_4/string_lookup_9/bincount/add/y:output:0*
T0	*
_output_shapes
: ?
1text_vectorization_4/string_lookup_9/bincount/mulMul6text_vectorization_4/string_lookup_9/bincount/Cast:y:05text_vectorization_4/string_lookup_9/bincount/add:z:0*
T0	*
_output_shapes
: z
7text_vectorization_4/string_lookup_9/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value
B	 R?N?
5text_vectorization_4/string_lookup_9/bincount/MaximumMaximum@text_vectorization_4/string_lookup_9/bincount/minlength:output:05text_vectorization_4/string_lookup_9/bincount/mul:z:0*
T0	*
_output_shapes
: z
7text_vectorization_4/string_lookup_9/bincount/maxlengthConst*
_output_shapes
: *
dtype0	*
value
B	 R?N?
5text_vectorization_4/string_lookup_9/bincount/MinimumMinimum@text_vectorization_4/string_lookup_9/bincount/maxlength:output:09text_vectorization_4/string_lookup_9/bincount/Maximum:z:0*
T0	*
_output_shapes
: x
5text_vectorization_4/string_lookup_9/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB ?
<text_vectorization_4/string_lookup_9/bincount/RaggedBincountRaggedBincountbtext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat:output:06text_vectorization_4/string_lookup_9/Identity:output:09text_vectorization_4/string_lookup_9/bincount/Minimum:z:0>text_vectorization_4/string_lookup_9/bincount/Const_2:output:0*
T0*

Tidx0	*(
_output_shapes
:??????????N?
 dense_24/StatefulPartitionedCallStatefulPartitionedCallEtext_vectorization_4/string_lookup_9/bincount/RaggedBincount:output:0dense_24_2465631dense_24_2465633*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_24_layer_call_and_return_conditional_losses_2465350?
"dropout_10/StatefulPartitionedCallStatefulPartitionedCall)dense_24/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dropout_10_layer_call_and_return_conditional_losses_2465522?
 dense_25/StatefulPartitionedCallStatefulPartitionedCall+dropout_10/StatefulPartitionedCall:output:0dense_25_2465637dense_25_2465639*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_25_layer_call_and_return_conditional_losses_2465374?
"dropout_11/StatefulPartitionedCallStatefulPartitionedCall)dense_25/StatefulPartitionedCall:output:0#^dropout_10/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dropout_11_layer_call_and_return_conditional_losses_2465489?
 dense_26/StatefulPartitionedCallStatefulPartitionedCall+dropout_11/StatefulPartitionedCall:output:0dense_26_2465643dense_26_2465645*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_26_layer_call_and_return_conditional_losses_2465398?
 dense_27/StatefulPartitionedCallStatefulPartitionedCall)dense_26/StatefulPartitionedCall:output:0dense_27_2465648dense_27_2465650*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_27_layer_call_and_return_conditional_losses_2465415x
IdentityIdentity)dense_27/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp!^dense_24/StatefulPartitionedCall!^dense_25/StatefulPartitionedCall!^dense_26/StatefulPartitionedCall!^dense_27/StatefulPartitionedCall#^dropout_10/StatefulPartitionedCall#^dropout_11/StatefulPartitionedCallC^text_vectorization_4/string_lookup_9/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????: : : : : : : : : : : : 2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall2D
 dense_25/StatefulPartitionedCall dense_25/StatefulPartitionedCall2D
 dense_26/StatefulPartitionedCall dense_26/StatefulPartitionedCall2D
 dense_27/StatefulPartitionedCall dense_27/StatefulPartitionedCall2H
"dropout_10/StatefulPartitionedCall"dropout_10/StatefulPartitionedCall2H
"dropout_11/StatefulPartitionedCall"dropout_11/StatefulPartitionedCall2?
Btext_vectorization_4/string_lookup_9/None_Lookup/LookupTableFindV2Btext_vectorization_4/string_lookup_9/None_Lookup/LookupTableFindV2:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
??
?
J__inference_ffnn_on_count_layer_call_and_return_conditional_losses_2465886
input_4S
Otext_vectorization_4_string_lookup_9_none_lookup_lookuptablefindv2_table_handleT
Ptext_vectorization_4_string_lookup_9_none_lookup_lookuptablefindv2_default_value	0
,text_vectorization_4_string_lookup_9_equal_y3
/text_vectorization_4_string_lookup_9_selectv2_t	#
dense_24_2465863:	?N
dense_24_2465865:"
dense_25_2465869:
dense_25_2465871:"
dense_26_2465875:

dense_26_2465877:
"
dense_27_2465880:

dense_27_2465882:
identity?? dense_24/StatefulPartitionedCall? dense_25/StatefulPartitionedCall? dense_26/StatefulPartitionedCall? dense_27/StatefulPartitionedCall?"dropout_10/StatefulPartitionedCall?"dropout_11/StatefulPartitionedCall?Btext_vectorization_4/string_lookup_9/None_Lookup/LookupTableFindV2a
 text_vectorization_4/StringLowerStringLowerinput_4*'
_output_shapes
:??????????
'text_vectorization_4/StaticRegexReplaceStaticRegexReplace)text_vectorization_4/StringLower:output:0*'
_output_shapes
:?????????*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite ?
text_vectorization_4/SqueezeSqueeze0text_vectorization_4/StaticRegexReplace:output:0*
T0*#
_output_shapes
:?????????*
squeeze_dims

?????????g
&text_vectorization_4/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
.text_vectorization_4/StringSplit/StringSplitV2StringSplitV2%text_vectorization_4/Squeeze:output:0/text_vectorization_4/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
4text_vectorization_4/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
6text_vectorization_4/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
6text_vectorization_4/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
.text_vectorization_4/StringSplit/strided_sliceStridedSlice8text_vectorization_4/StringSplit/StringSplitV2:indices:0=text_vectorization_4/StringSplit/strided_slice/stack:output:0?text_vectorization_4/StringSplit/strided_slice/stack_1:output:0?text_vectorization_4/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask?
6text_vectorization_4/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
8text_vectorization_4/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
8text_vectorization_4/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
0text_vectorization_4/StringSplit/strided_slice_1StridedSlice6text_vectorization_4/StringSplit/StringSplitV2:shape:0?text_vectorization_4/StringSplit/strided_slice_1/stack:output:0Atext_vectorization_4/StringSplit/strided_slice_1/stack_1:output:0Atext_vectorization_4/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Wtext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast7text_vectorization_4/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
Ytext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast9text_vectorization_4/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
atext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShape[text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
atext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
`text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdjtext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0jtext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
etext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
ctext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreateritext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ntext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
`text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastgtext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
ctext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
_text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMax[text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0ltext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
atext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
_text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2htext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0jtext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
_text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMuldtext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0ctext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum]text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0ctext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum]text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0gtext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
dtext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincount[text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0gtext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0ltext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
^text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ytext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumktext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0gtext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
btext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
^text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ytext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2ktext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0_text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0gtext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
Btext_vectorization_4/string_lookup_9/None_Lookup/LookupTableFindV2LookupTableFindV2Otext_vectorization_4_string_lookup_9_none_lookup_lookuptablefindv2_table_handle7text_vectorization_4/StringSplit/StringSplitV2:values:0Ptext_vectorization_4_string_lookup_9_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
*text_vectorization_4/string_lookup_9/EqualEqual7text_vectorization_4/StringSplit/StringSplitV2:values:0,text_vectorization_4_string_lookup_9_equal_y*
T0*#
_output_shapes
:??????????
-text_vectorization_4/string_lookup_9/SelectV2SelectV2.text_vectorization_4/string_lookup_9/Equal:z:0/text_vectorization_4_string_lookup_9_selectv2_tKtext_vectorization_4/string_lookup_9/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
-text_vectorization_4/string_lookup_9/IdentityIdentity6text_vectorization_4/string_lookup_9/SelectV2:output:0*
T0	*#
_output_shapes
:??????????
3text_vectorization_4/string_lookup_9/bincount/ShapeShape6text_vectorization_4/string_lookup_9/Identity:output:0*
T0	*
_output_shapes
:}
3text_vectorization_4/string_lookup_9/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
2text_vectorization_4/string_lookup_9/bincount/ProdProd<text_vectorization_4/string_lookup_9/bincount/Shape:output:0<text_vectorization_4/string_lookup_9/bincount/Const:output:0*
T0*
_output_shapes
: y
7text_vectorization_4/string_lookup_9/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
5text_vectorization_4/string_lookup_9/bincount/GreaterGreater;text_vectorization_4/string_lookup_9/bincount/Prod:output:0@text_vectorization_4/string_lookup_9/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
2text_vectorization_4/string_lookup_9/bincount/CastCast9text_vectorization_4/string_lookup_9/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 
5text_vectorization_4/string_lookup_9/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
=text_vectorization_4/string_lookup_9/bincount/RaggedReduceMaxMax6text_vectorization_4/string_lookup_9/Identity:output:0>text_vectorization_4/string_lookup_9/bincount/Const_1:output:0*
T0	*
_output_shapes
: u
3text_vectorization_4/string_lookup_9/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R?
1text_vectorization_4/string_lookup_9/bincount/addAddV2Ftext_vectorization_4/string_lookup_9/bincount/RaggedReduceMax:output:0<text_vectorization_4/string_lookup_9/bincount/add/y:output:0*
T0	*
_output_shapes
: ?
1text_vectorization_4/string_lookup_9/bincount/mulMul6text_vectorization_4/string_lookup_9/bincount/Cast:y:05text_vectorization_4/string_lookup_9/bincount/add:z:0*
T0	*
_output_shapes
: z
7text_vectorization_4/string_lookup_9/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value
B	 R?N?
5text_vectorization_4/string_lookup_9/bincount/MaximumMaximum@text_vectorization_4/string_lookup_9/bincount/minlength:output:05text_vectorization_4/string_lookup_9/bincount/mul:z:0*
T0	*
_output_shapes
: z
7text_vectorization_4/string_lookup_9/bincount/maxlengthConst*
_output_shapes
: *
dtype0	*
value
B	 R?N?
5text_vectorization_4/string_lookup_9/bincount/MinimumMinimum@text_vectorization_4/string_lookup_9/bincount/maxlength:output:09text_vectorization_4/string_lookup_9/bincount/Maximum:z:0*
T0	*
_output_shapes
: x
5text_vectorization_4/string_lookup_9/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB ?
<text_vectorization_4/string_lookup_9/bincount/RaggedBincountRaggedBincountbtext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat:output:06text_vectorization_4/string_lookup_9/Identity:output:09text_vectorization_4/string_lookup_9/bincount/Minimum:z:0>text_vectorization_4/string_lookup_9/bincount/Const_2:output:0*
T0*

Tidx0	*(
_output_shapes
:??????????N?
 dense_24/StatefulPartitionedCallStatefulPartitionedCallEtext_vectorization_4/string_lookup_9/bincount/RaggedBincount:output:0dense_24_2465863dense_24_2465865*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_24_layer_call_and_return_conditional_losses_2465350?
"dropout_10/StatefulPartitionedCallStatefulPartitionedCall)dense_24/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dropout_10_layer_call_and_return_conditional_losses_2465522?
 dense_25/StatefulPartitionedCallStatefulPartitionedCall+dropout_10/StatefulPartitionedCall:output:0dense_25_2465869dense_25_2465871*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_25_layer_call_and_return_conditional_losses_2465374?
"dropout_11/StatefulPartitionedCallStatefulPartitionedCall)dense_25/StatefulPartitionedCall:output:0#^dropout_10/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dropout_11_layer_call_and_return_conditional_losses_2465489?
 dense_26/StatefulPartitionedCallStatefulPartitionedCall+dropout_11/StatefulPartitionedCall:output:0dense_26_2465875dense_26_2465877*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_26_layer_call_and_return_conditional_losses_2465398?
 dense_27/StatefulPartitionedCallStatefulPartitionedCall)dense_26/StatefulPartitionedCall:output:0dense_27_2465880dense_27_2465882*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_27_layer_call_and_return_conditional_losses_2465415x
IdentityIdentity)dense_27/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp!^dense_24/StatefulPartitionedCall!^dense_25/StatefulPartitionedCall!^dense_26/StatefulPartitionedCall!^dense_27/StatefulPartitionedCall#^dropout_10/StatefulPartitionedCall#^dropout_11/StatefulPartitionedCallC^text_vectorization_4/string_lookup_9/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????: : : : : : : : : : : : 2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall2D
 dense_25/StatefulPartitionedCall dense_25/StatefulPartitionedCall2D
 dense_26/StatefulPartitionedCall dense_26/StatefulPartitionedCall2D
 dense_27/StatefulPartitionedCall dense_27/StatefulPartitionedCall2H
"dropout_10/StatefulPartitionedCall"dropout_10/StatefulPartitionedCall2H
"dropout_11/StatefulPartitionedCall"dropout_11/StatefulPartitionedCall2?
Btext_vectorization_4/string_lookup_9/None_Lookup/LookupTableFindV2Btext_vectorization_4/string_lookup_9/None_Lookup/LookupTableFindV2:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_4:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?	
f
G__inference_dropout_11_layer_call_and_return_conditional_losses_2466329

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
H
,__inference_dropout_10_layer_call_fn_2466260

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dropout_10_layer_call_and_return_conditional_losses_2465361`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
%__inference_signature_wrapper_2465923
input_4
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:	?N
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:

	unknown_8:

	unknown_9:


unknown_10:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *+
f&R$
"__inference__wrapped_model_2465270o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_4:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
H
__inference__creator_2466392
identity: ??MutableHashTable?
MutableHashTableMutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_nametable_2016626*
value_dtype0	]
IdentityIdentityMutableHashTable:table_handle:0^NoOp*
T0*
_output_shapes
: Y
NoOpNoOp^MutableHashTable*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2$
MutableHashTableMutableHashTable
?
e
G__inference_dropout_11_layer_call_and_return_conditional_losses_2465385

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
/__inference_ffnn_on_count_layer_call_fn_2466029

inputs
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:	?N
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:

	unknown_8:

	unknown_9:


unknown_10:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_ffnn_on_count_layer_call_and_return_conditional_losses_2465654o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
??
?
J__inference_ffnn_on_count_layer_call_and_return_conditional_losses_2465422

inputsS
Otext_vectorization_4_string_lookup_9_none_lookup_lookuptablefindv2_table_handleT
Ptext_vectorization_4_string_lookup_9_none_lookup_lookuptablefindv2_default_value	0
,text_vectorization_4_string_lookup_9_equal_y3
/text_vectorization_4_string_lookup_9_selectv2_t	#
dense_24_2465351:	?N
dense_24_2465353:"
dense_25_2465375:
dense_25_2465377:"
dense_26_2465399:

dense_26_2465401:
"
dense_27_2465416:

dense_27_2465418:
identity?? dense_24/StatefulPartitionedCall? dense_25/StatefulPartitionedCall? dense_26/StatefulPartitionedCall? dense_27/StatefulPartitionedCall?Btext_vectorization_4/string_lookup_9/None_Lookup/LookupTableFindV2`
 text_vectorization_4/StringLowerStringLowerinputs*'
_output_shapes
:??????????
'text_vectorization_4/StaticRegexReplaceStaticRegexReplace)text_vectorization_4/StringLower:output:0*'
_output_shapes
:?????????*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite ?
text_vectorization_4/SqueezeSqueeze0text_vectorization_4/StaticRegexReplace:output:0*
T0*#
_output_shapes
:?????????*
squeeze_dims

?????????g
&text_vectorization_4/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
.text_vectorization_4/StringSplit/StringSplitV2StringSplitV2%text_vectorization_4/Squeeze:output:0/text_vectorization_4/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
4text_vectorization_4/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
6text_vectorization_4/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
6text_vectorization_4/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
.text_vectorization_4/StringSplit/strided_sliceStridedSlice8text_vectorization_4/StringSplit/StringSplitV2:indices:0=text_vectorization_4/StringSplit/strided_slice/stack:output:0?text_vectorization_4/StringSplit/strided_slice/stack_1:output:0?text_vectorization_4/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask?
6text_vectorization_4/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
8text_vectorization_4/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
8text_vectorization_4/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
0text_vectorization_4/StringSplit/strided_slice_1StridedSlice6text_vectorization_4/StringSplit/StringSplitV2:shape:0?text_vectorization_4/StringSplit/strided_slice_1/stack:output:0Atext_vectorization_4/StringSplit/strided_slice_1/stack_1:output:0Atext_vectorization_4/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Wtext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast7text_vectorization_4/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
Ytext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast9text_vectorization_4/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
atext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShape[text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
atext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
`text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdjtext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0jtext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
etext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
ctext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreateritext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ntext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
`text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastgtext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
ctext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
_text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMax[text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0ltext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
atext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
_text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2htext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0jtext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
_text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMuldtext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0ctext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum]text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0ctext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum]text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0gtext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
dtext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincount[text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0gtext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0ltext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
^text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ytext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumktext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0gtext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
btext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
^text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ytext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2ktext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0_text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0gtext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
Btext_vectorization_4/string_lookup_9/None_Lookup/LookupTableFindV2LookupTableFindV2Otext_vectorization_4_string_lookup_9_none_lookup_lookuptablefindv2_table_handle7text_vectorization_4/StringSplit/StringSplitV2:values:0Ptext_vectorization_4_string_lookup_9_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
*text_vectorization_4/string_lookup_9/EqualEqual7text_vectorization_4/StringSplit/StringSplitV2:values:0,text_vectorization_4_string_lookup_9_equal_y*
T0*#
_output_shapes
:??????????
-text_vectorization_4/string_lookup_9/SelectV2SelectV2.text_vectorization_4/string_lookup_9/Equal:z:0/text_vectorization_4_string_lookup_9_selectv2_tKtext_vectorization_4/string_lookup_9/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
-text_vectorization_4/string_lookup_9/IdentityIdentity6text_vectorization_4/string_lookup_9/SelectV2:output:0*
T0	*#
_output_shapes
:??????????
3text_vectorization_4/string_lookup_9/bincount/ShapeShape6text_vectorization_4/string_lookup_9/Identity:output:0*
T0	*
_output_shapes
:}
3text_vectorization_4/string_lookup_9/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
2text_vectorization_4/string_lookup_9/bincount/ProdProd<text_vectorization_4/string_lookup_9/bincount/Shape:output:0<text_vectorization_4/string_lookup_9/bincount/Const:output:0*
T0*
_output_shapes
: y
7text_vectorization_4/string_lookup_9/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
5text_vectorization_4/string_lookup_9/bincount/GreaterGreater;text_vectorization_4/string_lookup_9/bincount/Prod:output:0@text_vectorization_4/string_lookup_9/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
2text_vectorization_4/string_lookup_9/bincount/CastCast9text_vectorization_4/string_lookup_9/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 
5text_vectorization_4/string_lookup_9/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
=text_vectorization_4/string_lookup_9/bincount/RaggedReduceMaxMax6text_vectorization_4/string_lookup_9/Identity:output:0>text_vectorization_4/string_lookup_9/bincount/Const_1:output:0*
T0	*
_output_shapes
: u
3text_vectorization_4/string_lookup_9/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R?
1text_vectorization_4/string_lookup_9/bincount/addAddV2Ftext_vectorization_4/string_lookup_9/bincount/RaggedReduceMax:output:0<text_vectorization_4/string_lookup_9/bincount/add/y:output:0*
T0	*
_output_shapes
: ?
1text_vectorization_4/string_lookup_9/bincount/mulMul6text_vectorization_4/string_lookup_9/bincount/Cast:y:05text_vectorization_4/string_lookup_9/bincount/add:z:0*
T0	*
_output_shapes
: z
7text_vectorization_4/string_lookup_9/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value
B	 R?N?
5text_vectorization_4/string_lookup_9/bincount/MaximumMaximum@text_vectorization_4/string_lookup_9/bincount/minlength:output:05text_vectorization_4/string_lookup_9/bincount/mul:z:0*
T0	*
_output_shapes
: z
7text_vectorization_4/string_lookup_9/bincount/maxlengthConst*
_output_shapes
: *
dtype0	*
value
B	 R?N?
5text_vectorization_4/string_lookup_9/bincount/MinimumMinimum@text_vectorization_4/string_lookup_9/bincount/maxlength:output:09text_vectorization_4/string_lookup_9/bincount/Maximum:z:0*
T0	*
_output_shapes
: x
5text_vectorization_4/string_lookup_9/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB ?
<text_vectorization_4/string_lookup_9/bincount/RaggedBincountRaggedBincountbtext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat:output:06text_vectorization_4/string_lookup_9/Identity:output:09text_vectorization_4/string_lookup_9/bincount/Minimum:z:0>text_vectorization_4/string_lookup_9/bincount/Const_2:output:0*
T0*

Tidx0	*(
_output_shapes
:??????????N?
 dense_24/StatefulPartitionedCallStatefulPartitionedCallEtext_vectorization_4/string_lookup_9/bincount/RaggedBincount:output:0dense_24_2465351dense_24_2465353*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_24_layer_call_and_return_conditional_losses_2465350?
dropout_10/PartitionedCallPartitionedCall)dense_24/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dropout_10_layer_call_and_return_conditional_losses_2465361?
 dense_25/StatefulPartitionedCallStatefulPartitionedCall#dropout_10/PartitionedCall:output:0dense_25_2465375dense_25_2465377*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_25_layer_call_and_return_conditional_losses_2465374?
dropout_11/PartitionedCallPartitionedCall)dense_25/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dropout_11_layer_call_and_return_conditional_losses_2465385?
 dense_26/StatefulPartitionedCallStatefulPartitionedCall#dropout_11/PartitionedCall:output:0dense_26_2465399dense_26_2465401*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_26_layer_call_and_return_conditional_losses_2465398?
 dense_27/StatefulPartitionedCallStatefulPartitionedCall)dense_26/StatefulPartitionedCall:output:0dense_27_2465416dense_27_2465418*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_27_layer_call_and_return_conditional_losses_2465415x
IdentityIdentity)dense_27/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp!^dense_24/StatefulPartitionedCall!^dense_25/StatefulPartitionedCall!^dense_26/StatefulPartitionedCall!^dense_27/StatefulPartitionedCallC^text_vectorization_4/string_lookup_9/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????: : : : : : : : : : : : 2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall2D
 dense_25/StatefulPartitionedCall dense_25/StatefulPartitionedCall2D
 dense_26/StatefulPartitionedCall dense_26/StatefulPartitionedCall2D
 dense_27/StatefulPartitionedCall dense_27/StatefulPartitionedCall2?
Btext_vectorization_4/string_lookup_9/None_Lookup/LookupTableFindV2Btext_vectorization_4/string_lookup_9/None_Lookup/LookupTableFindV2:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
??
?
"__inference__wrapped_model_2465270
input_4a
]ffnn_on_count_text_vectorization_4_string_lookup_9_none_lookup_lookuptablefindv2_table_handleb
^ffnn_on_count_text_vectorization_4_string_lookup_9_none_lookup_lookuptablefindv2_default_value	>
:ffnn_on_count_text_vectorization_4_string_lookup_9_equal_yA
=ffnn_on_count_text_vectorization_4_string_lookup_9_selectv2_t	H
5ffnn_on_count_dense_24_matmul_readvariableop_resource:	?ND
6ffnn_on_count_dense_24_biasadd_readvariableop_resource:G
5ffnn_on_count_dense_25_matmul_readvariableop_resource:D
6ffnn_on_count_dense_25_biasadd_readvariableop_resource:G
5ffnn_on_count_dense_26_matmul_readvariableop_resource:
D
6ffnn_on_count_dense_26_biasadd_readvariableop_resource:
G
5ffnn_on_count_dense_27_matmul_readvariableop_resource:
D
6ffnn_on_count_dense_27_biasadd_readvariableop_resource:
identity??-ffnn_on_count/dense_24/BiasAdd/ReadVariableOp?,ffnn_on_count/dense_24/MatMul/ReadVariableOp?-ffnn_on_count/dense_25/BiasAdd/ReadVariableOp?,ffnn_on_count/dense_25/MatMul/ReadVariableOp?-ffnn_on_count/dense_26/BiasAdd/ReadVariableOp?,ffnn_on_count/dense_26/MatMul/ReadVariableOp?-ffnn_on_count/dense_27/BiasAdd/ReadVariableOp?,ffnn_on_count/dense_27/MatMul/ReadVariableOp?Pffnn_on_count/text_vectorization_4/string_lookup_9/None_Lookup/LookupTableFindV2o
.ffnn_on_count/text_vectorization_4/StringLowerStringLowerinput_4*'
_output_shapes
:??????????
5ffnn_on_count/text_vectorization_4/StaticRegexReplaceStaticRegexReplace7ffnn_on_count/text_vectorization_4/StringLower:output:0*'
_output_shapes
:?????????*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite ?
*ffnn_on_count/text_vectorization_4/SqueezeSqueeze>ffnn_on_count/text_vectorization_4/StaticRegexReplace:output:0*
T0*#
_output_shapes
:?????????*
squeeze_dims

?????????u
4ffnn_on_count/text_vectorization_4/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
<ffnn_on_count/text_vectorization_4/StringSplit/StringSplitV2StringSplitV23ffnn_on_count/text_vectorization_4/Squeeze:output:0=ffnn_on_count/text_vectorization_4/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
Bffnn_on_count/text_vectorization_4/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
Dffnn_on_count/text_vectorization_4/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
Dffnn_on_count/text_vectorization_4/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
<ffnn_on_count/text_vectorization_4/StringSplit/strided_sliceStridedSliceFffnn_on_count/text_vectorization_4/StringSplit/StringSplitV2:indices:0Kffnn_on_count/text_vectorization_4/StringSplit/strided_slice/stack:output:0Mffnn_on_count/text_vectorization_4/StringSplit/strided_slice/stack_1:output:0Mffnn_on_count/text_vectorization_4/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask?
Dffnn_on_count/text_vectorization_4/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Fffnn_on_count/text_vectorization_4/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Fffnn_on_count/text_vectorization_4/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
>ffnn_on_count/text_vectorization_4/StringSplit/strided_slice_1StridedSliceDffnn_on_count/text_vectorization_4/StringSplit/StringSplitV2:shape:0Mffnn_on_count/text_vectorization_4/StringSplit/strided_slice_1/stack:output:0Offnn_on_count/text_vectorization_4/StringSplit/strided_slice_1/stack_1:output:0Offnn_on_count/text_vectorization_4/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
effnn_on_count/text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCastEffnn_on_count/text_vectorization_4/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
gffnn_on_count/text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1CastGffnn_on_count/text_vectorization_4/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
offnn_on_count/text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeiffnn_on_count/text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
offnn_on_count/text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
nffnn_on_count/text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdxffnn_on_count/text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0xffnn_on_count/text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
sffnn_on_count/text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
qffnn_on_count/text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreaterwffnn_on_count/text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0|ffnn_on_count/text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
nffnn_on_count/text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastuffnn_on_count/text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
qffnn_on_count/text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
mffnn_on_count/text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxiffnn_on_count/text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0zffnn_on_count/text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
offnn_on_count/text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
mffnn_on_count/text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2vffnn_on_count/text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0xffnn_on_count/text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
mffnn_on_count/text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulrffnn_on_count/text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0qffnn_on_count/text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
qffnn_on_count/text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximumkffnn_on_count/text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0qffnn_on_count/text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
qffnn_on_count/text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimumkffnn_on_count/text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0uffnn_on_count/text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
qffnn_on_count/text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
rffnn_on_count/text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountiffnn_on_count/text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0uffnn_on_count/text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0zffnn_on_count/text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
lffnn_on_count/text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
gffnn_on_count/text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumyffnn_on_count/text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0uffnn_on_count/text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
pffnn_on_count/text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
lffnn_on_count/text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
gffnn_on_count/text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2yffnn_on_count/text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0mffnn_on_count/text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0uffnn_on_count/text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
Pffnn_on_count/text_vectorization_4/string_lookup_9/None_Lookup/LookupTableFindV2LookupTableFindV2]ffnn_on_count_text_vectorization_4_string_lookup_9_none_lookup_lookuptablefindv2_table_handleEffnn_on_count/text_vectorization_4/StringSplit/StringSplitV2:values:0^ffnn_on_count_text_vectorization_4_string_lookup_9_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
8ffnn_on_count/text_vectorization_4/string_lookup_9/EqualEqualEffnn_on_count/text_vectorization_4/StringSplit/StringSplitV2:values:0:ffnn_on_count_text_vectorization_4_string_lookup_9_equal_y*
T0*#
_output_shapes
:??????????
;ffnn_on_count/text_vectorization_4/string_lookup_9/SelectV2SelectV2<ffnn_on_count/text_vectorization_4/string_lookup_9/Equal:z:0=ffnn_on_count_text_vectorization_4_string_lookup_9_selectv2_tYffnn_on_count/text_vectorization_4/string_lookup_9/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
;ffnn_on_count/text_vectorization_4/string_lookup_9/IdentityIdentityDffnn_on_count/text_vectorization_4/string_lookup_9/SelectV2:output:0*
T0	*#
_output_shapes
:??????????
Affnn_on_count/text_vectorization_4/string_lookup_9/bincount/ShapeShapeDffnn_on_count/text_vectorization_4/string_lookup_9/Identity:output:0*
T0	*
_output_shapes
:?
Affnn_on_count/text_vectorization_4/string_lookup_9/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
@ffnn_on_count/text_vectorization_4/string_lookup_9/bincount/ProdProdJffnn_on_count/text_vectorization_4/string_lookup_9/bincount/Shape:output:0Jffnn_on_count/text_vectorization_4/string_lookup_9/bincount/Const:output:0*
T0*
_output_shapes
: ?
Effnn_on_count/text_vectorization_4/string_lookup_9/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
Cffnn_on_count/text_vectorization_4/string_lookup_9/bincount/GreaterGreaterIffnn_on_count/text_vectorization_4/string_lookup_9/bincount/Prod:output:0Nffnn_on_count/text_vectorization_4/string_lookup_9/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
@ffnn_on_count/text_vectorization_4/string_lookup_9/bincount/CastCastGffnn_on_count/text_vectorization_4/string_lookup_9/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: ?
Cffnn_on_count/text_vectorization_4/string_lookup_9/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
Kffnn_on_count/text_vectorization_4/string_lookup_9/bincount/RaggedReduceMaxMaxDffnn_on_count/text_vectorization_4/string_lookup_9/Identity:output:0Lffnn_on_count/text_vectorization_4/string_lookup_9/bincount/Const_1:output:0*
T0	*
_output_shapes
: ?
Affnn_on_count/text_vectorization_4/string_lookup_9/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R?
?ffnn_on_count/text_vectorization_4/string_lookup_9/bincount/addAddV2Tffnn_on_count/text_vectorization_4/string_lookup_9/bincount/RaggedReduceMax:output:0Jffnn_on_count/text_vectorization_4/string_lookup_9/bincount/add/y:output:0*
T0	*
_output_shapes
: ?
?ffnn_on_count/text_vectorization_4/string_lookup_9/bincount/mulMulDffnn_on_count/text_vectorization_4/string_lookup_9/bincount/Cast:y:0Cffnn_on_count/text_vectorization_4/string_lookup_9/bincount/add:z:0*
T0	*
_output_shapes
: ?
Effnn_on_count/text_vectorization_4/string_lookup_9/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value
B	 R?N?
Cffnn_on_count/text_vectorization_4/string_lookup_9/bincount/MaximumMaximumNffnn_on_count/text_vectorization_4/string_lookup_9/bincount/minlength:output:0Cffnn_on_count/text_vectorization_4/string_lookup_9/bincount/mul:z:0*
T0	*
_output_shapes
: ?
Effnn_on_count/text_vectorization_4/string_lookup_9/bincount/maxlengthConst*
_output_shapes
: *
dtype0	*
value
B	 R?N?
Cffnn_on_count/text_vectorization_4/string_lookup_9/bincount/MinimumMinimumNffnn_on_count/text_vectorization_4/string_lookup_9/bincount/maxlength:output:0Gffnn_on_count/text_vectorization_4/string_lookup_9/bincount/Maximum:z:0*
T0	*
_output_shapes
: ?
Cffnn_on_count/text_vectorization_4/string_lookup_9/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB ?
Jffnn_on_count/text_vectorization_4/string_lookup_9/bincount/RaggedBincountRaggedBincountpffnn_on_count/text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat:output:0Dffnn_on_count/text_vectorization_4/string_lookup_9/Identity:output:0Gffnn_on_count/text_vectorization_4/string_lookup_9/bincount/Minimum:z:0Lffnn_on_count/text_vectorization_4/string_lookup_9/bincount/Const_2:output:0*
T0*

Tidx0	*(
_output_shapes
:??????????N?
,ffnn_on_count/dense_24/MatMul/ReadVariableOpReadVariableOp5ffnn_on_count_dense_24_matmul_readvariableop_resource*
_output_shapes
:	?N*
dtype0?
ffnn_on_count/dense_24/MatMulMatMulSffnn_on_count/text_vectorization_4/string_lookup_9/bincount/RaggedBincount:output:04ffnn_on_count/dense_24/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
-ffnn_on_count/dense_24/BiasAdd/ReadVariableOpReadVariableOp6ffnn_on_count_dense_24_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
ffnn_on_count/dense_24/BiasAddBiasAdd'ffnn_on_count/dense_24/MatMul:product:05ffnn_on_count/dense_24/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????~
ffnn_on_count/dense_24/ReluRelu'ffnn_on_count/dense_24/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
!ffnn_on_count/dropout_10/IdentityIdentity)ffnn_on_count/dense_24/Relu:activations:0*
T0*'
_output_shapes
:??????????
,ffnn_on_count/dense_25/MatMul/ReadVariableOpReadVariableOp5ffnn_on_count_dense_25_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
ffnn_on_count/dense_25/MatMulMatMul*ffnn_on_count/dropout_10/Identity:output:04ffnn_on_count/dense_25/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
-ffnn_on_count/dense_25/BiasAdd/ReadVariableOpReadVariableOp6ffnn_on_count_dense_25_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
ffnn_on_count/dense_25/BiasAddBiasAdd'ffnn_on_count/dense_25/MatMul:product:05ffnn_on_count/dense_25/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????~
ffnn_on_count/dense_25/ReluRelu'ffnn_on_count/dense_25/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
!ffnn_on_count/dropout_11/IdentityIdentity)ffnn_on_count/dense_25/Relu:activations:0*
T0*'
_output_shapes
:??????????
,ffnn_on_count/dense_26/MatMul/ReadVariableOpReadVariableOp5ffnn_on_count_dense_26_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0?
ffnn_on_count/dense_26/MatMulMatMul*ffnn_on_count/dropout_11/Identity:output:04ffnn_on_count/dense_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
-ffnn_on_count/dense_26/BiasAdd/ReadVariableOpReadVariableOp6ffnn_on_count_dense_26_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0?
ffnn_on_count/dense_26/BiasAddBiasAdd'ffnn_on_count/dense_26/MatMul:product:05ffnn_on_count/dense_26/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
~
ffnn_on_count/dense_26/ReluRelu'ffnn_on_count/dense_26/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
?
,ffnn_on_count/dense_27/MatMul/ReadVariableOpReadVariableOp5ffnn_on_count_dense_27_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0?
ffnn_on_count/dense_27/MatMulMatMul)ffnn_on_count/dense_26/Relu:activations:04ffnn_on_count/dense_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
-ffnn_on_count/dense_27/BiasAdd/ReadVariableOpReadVariableOp6ffnn_on_count_dense_27_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
ffnn_on_count/dense_27/BiasAddBiasAdd'ffnn_on_count/dense_27/MatMul:product:05ffnn_on_count/dense_27/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
ffnn_on_count/dense_27/SigmoidSigmoid'ffnn_on_count/dense_27/BiasAdd:output:0*
T0*'
_output_shapes
:?????????q
IdentityIdentity"ffnn_on_count/dense_27/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp.^ffnn_on_count/dense_24/BiasAdd/ReadVariableOp-^ffnn_on_count/dense_24/MatMul/ReadVariableOp.^ffnn_on_count/dense_25/BiasAdd/ReadVariableOp-^ffnn_on_count/dense_25/MatMul/ReadVariableOp.^ffnn_on_count/dense_26/BiasAdd/ReadVariableOp-^ffnn_on_count/dense_26/MatMul/ReadVariableOp.^ffnn_on_count/dense_27/BiasAdd/ReadVariableOp-^ffnn_on_count/dense_27/MatMul/ReadVariableOpQ^ffnn_on_count/text_vectorization_4/string_lookup_9/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????: : : : : : : : : : : : 2^
-ffnn_on_count/dense_24/BiasAdd/ReadVariableOp-ffnn_on_count/dense_24/BiasAdd/ReadVariableOp2\
,ffnn_on_count/dense_24/MatMul/ReadVariableOp,ffnn_on_count/dense_24/MatMul/ReadVariableOp2^
-ffnn_on_count/dense_25/BiasAdd/ReadVariableOp-ffnn_on_count/dense_25/BiasAdd/ReadVariableOp2\
,ffnn_on_count/dense_25/MatMul/ReadVariableOp,ffnn_on_count/dense_25/MatMul/ReadVariableOp2^
-ffnn_on_count/dense_26/BiasAdd/ReadVariableOp-ffnn_on_count/dense_26/BiasAdd/ReadVariableOp2\
,ffnn_on_count/dense_26/MatMul/ReadVariableOp,ffnn_on_count/dense_26/MatMul/ReadVariableOp2^
-ffnn_on_count/dense_27/BiasAdd/ReadVariableOp-ffnn_on_count/dense_27/BiasAdd/ReadVariableOp2\
,ffnn_on_count/dense_27/MatMul/ReadVariableOp,ffnn_on_count/dense_27/MatMul/ReadVariableOp2?
Pffnn_on_count/text_vectorization_4/string_lookup_9/None_Lookup/LookupTableFindV2Pffnn_on_count/text_vectorization_4/string_lookup_9/None_Lookup/LookupTableFindV2:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_4:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
,
__inference_<lambda>_2466442
identityJ
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?

?
E__inference_dense_25_layer_call_and_return_conditional_losses_2466302

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?	
J__inference_ffnn_on_count_layer_call_and_return_conditional_losses_2466235

inputsS
Otext_vectorization_4_string_lookup_9_none_lookup_lookuptablefindv2_table_handleT
Ptext_vectorization_4_string_lookup_9_none_lookup_lookuptablefindv2_default_value	0
,text_vectorization_4_string_lookup_9_equal_y3
/text_vectorization_4_string_lookup_9_selectv2_t	:
'dense_24_matmul_readvariableop_resource:	?N6
(dense_24_biasadd_readvariableop_resource:9
'dense_25_matmul_readvariableop_resource:6
(dense_25_biasadd_readvariableop_resource:9
'dense_26_matmul_readvariableop_resource:
6
(dense_26_biasadd_readvariableop_resource:
9
'dense_27_matmul_readvariableop_resource:
6
(dense_27_biasadd_readvariableop_resource:
identity??dense_24/BiasAdd/ReadVariableOp?dense_24/MatMul/ReadVariableOp?dense_25/BiasAdd/ReadVariableOp?dense_25/MatMul/ReadVariableOp?dense_26/BiasAdd/ReadVariableOp?dense_26/MatMul/ReadVariableOp?dense_27/BiasAdd/ReadVariableOp?dense_27/MatMul/ReadVariableOp?Btext_vectorization_4/string_lookup_9/None_Lookup/LookupTableFindV2`
 text_vectorization_4/StringLowerStringLowerinputs*'
_output_shapes
:??????????
'text_vectorization_4/StaticRegexReplaceStaticRegexReplace)text_vectorization_4/StringLower:output:0*'
_output_shapes
:?????????*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite ?
text_vectorization_4/SqueezeSqueeze0text_vectorization_4/StaticRegexReplace:output:0*
T0*#
_output_shapes
:?????????*
squeeze_dims

?????????g
&text_vectorization_4/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
.text_vectorization_4/StringSplit/StringSplitV2StringSplitV2%text_vectorization_4/Squeeze:output:0/text_vectorization_4/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
4text_vectorization_4/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
6text_vectorization_4/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
6text_vectorization_4/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
.text_vectorization_4/StringSplit/strided_sliceStridedSlice8text_vectorization_4/StringSplit/StringSplitV2:indices:0=text_vectorization_4/StringSplit/strided_slice/stack:output:0?text_vectorization_4/StringSplit/strided_slice/stack_1:output:0?text_vectorization_4/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask?
6text_vectorization_4/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
8text_vectorization_4/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
8text_vectorization_4/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
0text_vectorization_4/StringSplit/strided_slice_1StridedSlice6text_vectorization_4/StringSplit/StringSplitV2:shape:0?text_vectorization_4/StringSplit/strided_slice_1/stack:output:0Atext_vectorization_4/StringSplit/strided_slice_1/stack_1:output:0Atext_vectorization_4/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Wtext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast7text_vectorization_4/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
Ytext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast9text_vectorization_4/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
atext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShape[text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
atext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
`text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdjtext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0jtext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
etext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
ctext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreateritext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ntext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
`text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastgtext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
ctext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
_text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMax[text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0ltext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
atext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
_text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2htext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0jtext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
_text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMuldtext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0ctext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum]text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0ctext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum]text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0gtext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
dtext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincount[text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0gtext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0ltext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
^text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ytext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumktext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0gtext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
btext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
^text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ytext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2ktext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0_text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0gtext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
Btext_vectorization_4/string_lookup_9/None_Lookup/LookupTableFindV2LookupTableFindV2Otext_vectorization_4_string_lookup_9_none_lookup_lookuptablefindv2_table_handle7text_vectorization_4/StringSplit/StringSplitV2:values:0Ptext_vectorization_4_string_lookup_9_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
*text_vectorization_4/string_lookup_9/EqualEqual7text_vectorization_4/StringSplit/StringSplitV2:values:0,text_vectorization_4_string_lookup_9_equal_y*
T0*#
_output_shapes
:??????????
-text_vectorization_4/string_lookup_9/SelectV2SelectV2.text_vectorization_4/string_lookup_9/Equal:z:0/text_vectorization_4_string_lookup_9_selectv2_tKtext_vectorization_4/string_lookup_9/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
-text_vectorization_4/string_lookup_9/IdentityIdentity6text_vectorization_4/string_lookup_9/SelectV2:output:0*
T0	*#
_output_shapes
:??????????
3text_vectorization_4/string_lookup_9/bincount/ShapeShape6text_vectorization_4/string_lookup_9/Identity:output:0*
T0	*
_output_shapes
:}
3text_vectorization_4/string_lookup_9/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
2text_vectorization_4/string_lookup_9/bincount/ProdProd<text_vectorization_4/string_lookup_9/bincount/Shape:output:0<text_vectorization_4/string_lookup_9/bincount/Const:output:0*
T0*
_output_shapes
: y
7text_vectorization_4/string_lookup_9/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
5text_vectorization_4/string_lookup_9/bincount/GreaterGreater;text_vectorization_4/string_lookup_9/bincount/Prod:output:0@text_vectorization_4/string_lookup_9/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
2text_vectorization_4/string_lookup_9/bincount/CastCast9text_vectorization_4/string_lookup_9/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 
5text_vectorization_4/string_lookup_9/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
=text_vectorization_4/string_lookup_9/bincount/RaggedReduceMaxMax6text_vectorization_4/string_lookup_9/Identity:output:0>text_vectorization_4/string_lookup_9/bincount/Const_1:output:0*
T0	*
_output_shapes
: u
3text_vectorization_4/string_lookup_9/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R?
1text_vectorization_4/string_lookup_9/bincount/addAddV2Ftext_vectorization_4/string_lookup_9/bincount/RaggedReduceMax:output:0<text_vectorization_4/string_lookup_9/bincount/add/y:output:0*
T0	*
_output_shapes
: ?
1text_vectorization_4/string_lookup_9/bincount/mulMul6text_vectorization_4/string_lookup_9/bincount/Cast:y:05text_vectorization_4/string_lookup_9/bincount/add:z:0*
T0	*
_output_shapes
: z
7text_vectorization_4/string_lookup_9/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value
B	 R?N?
5text_vectorization_4/string_lookup_9/bincount/MaximumMaximum@text_vectorization_4/string_lookup_9/bincount/minlength:output:05text_vectorization_4/string_lookup_9/bincount/mul:z:0*
T0	*
_output_shapes
: z
7text_vectorization_4/string_lookup_9/bincount/maxlengthConst*
_output_shapes
: *
dtype0	*
value
B	 R?N?
5text_vectorization_4/string_lookup_9/bincount/MinimumMinimum@text_vectorization_4/string_lookup_9/bincount/maxlength:output:09text_vectorization_4/string_lookup_9/bincount/Maximum:z:0*
T0	*
_output_shapes
: x
5text_vectorization_4/string_lookup_9/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB ?
<text_vectorization_4/string_lookup_9/bincount/RaggedBincountRaggedBincountbtext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat:output:06text_vectorization_4/string_lookup_9/Identity:output:09text_vectorization_4/string_lookup_9/bincount/Minimum:z:0>text_vectorization_4/string_lookup_9/bincount/Const_2:output:0*
T0*

Tidx0	*(
_output_shapes
:??????????N?
dense_24/MatMul/ReadVariableOpReadVariableOp'dense_24_matmul_readvariableop_resource*
_output_shapes
:	?N*
dtype0?
dense_24/MatMulMatMulEtext_vectorization_4/string_lookup_9/bincount/RaggedBincount:output:0&dense_24/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_24/BiasAdd/ReadVariableOpReadVariableOp(dense_24_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_24/BiasAddBiasAdddense_24/MatMul:product:0'dense_24/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????b
dense_24/ReluReludense_24/BiasAdd:output:0*
T0*'
_output_shapes
:?????????]
dropout_10/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶??
dropout_10/dropout/MulMuldense_24/Relu:activations:0!dropout_10/dropout/Const:output:0*
T0*'
_output_shapes
:?????????c
dropout_10/dropout/ShapeShapedense_24/Relu:activations:0*
T0*
_output_shapes
:?
/dropout_10/dropout/random_uniform/RandomUniformRandomUniform!dropout_10/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0f
!dropout_10/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>?
dropout_10/dropout/GreaterEqualGreaterEqual8dropout_10/dropout/random_uniform/RandomUniform:output:0*dropout_10/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:??????????
dropout_10/dropout/CastCast#dropout_10/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:??????????
dropout_10/dropout/Mul_1Muldropout_10/dropout/Mul:z:0dropout_10/dropout/Cast:y:0*
T0*'
_output_shapes
:??????????
dense_25/MatMul/ReadVariableOpReadVariableOp'dense_25_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
dense_25/MatMulMatMuldropout_10/dropout/Mul_1:z:0&dense_25/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_25/BiasAdd/ReadVariableOpReadVariableOp(dense_25_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_25/BiasAddBiasAdddense_25/MatMul:product:0'dense_25/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????b
dense_25/ReluReludense_25/BiasAdd:output:0*
T0*'
_output_shapes
:?????????]
dropout_11/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶??
dropout_11/dropout/MulMuldense_25/Relu:activations:0!dropout_11/dropout/Const:output:0*
T0*'
_output_shapes
:?????????c
dropout_11/dropout/ShapeShapedense_25/Relu:activations:0*
T0*
_output_shapes
:?
/dropout_11/dropout/random_uniform/RandomUniformRandomUniform!dropout_11/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0f
!dropout_11/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>?
dropout_11/dropout/GreaterEqualGreaterEqual8dropout_11/dropout/random_uniform/RandomUniform:output:0*dropout_11/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:??????????
dropout_11/dropout/CastCast#dropout_11/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:??????????
dropout_11/dropout/Mul_1Muldropout_11/dropout/Mul:z:0dropout_11/dropout/Cast:y:0*
T0*'
_output_shapes
:??????????
dense_26/MatMul/ReadVariableOpReadVariableOp'dense_26_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0?
dense_26/MatMulMatMuldropout_11/dropout/Mul_1:z:0&dense_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
dense_26/BiasAdd/ReadVariableOpReadVariableOp(dense_26_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0?
dense_26/BiasAddBiasAdddense_26/MatMul:product:0'dense_26/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
b
dense_26/ReluReludense_26/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
?
dense_27/MatMul/ReadVariableOpReadVariableOp'dense_27_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0?
dense_27/MatMulMatMuldense_26/Relu:activations:0&dense_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_27/BiasAdd/ReadVariableOpReadVariableOp(dense_27_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_27/BiasAddBiasAdddense_27/MatMul:product:0'dense_27/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????h
dense_27/SigmoidSigmoiddense_27/BiasAdd:output:0*
T0*'
_output_shapes
:?????????c
IdentityIdentitydense_27/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_24/BiasAdd/ReadVariableOp^dense_24/MatMul/ReadVariableOp ^dense_25/BiasAdd/ReadVariableOp^dense_25/MatMul/ReadVariableOp ^dense_26/BiasAdd/ReadVariableOp^dense_26/MatMul/ReadVariableOp ^dense_27/BiasAdd/ReadVariableOp^dense_27/MatMul/ReadVariableOpC^text_vectorization_4/string_lookup_9/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????: : : : : : : : : : : : 2B
dense_24/BiasAdd/ReadVariableOpdense_24/BiasAdd/ReadVariableOp2@
dense_24/MatMul/ReadVariableOpdense_24/MatMul/ReadVariableOp2B
dense_25/BiasAdd/ReadVariableOpdense_25/BiasAdd/ReadVariableOp2@
dense_25/MatMul/ReadVariableOpdense_25/MatMul/ReadVariableOp2B
dense_26/BiasAdd/ReadVariableOpdense_26/BiasAdd/ReadVariableOp2@
dense_26/MatMul/ReadVariableOpdense_26/MatMul/ReadVariableOp2B
dense_27/BiasAdd/ReadVariableOpdense_27/BiasAdd/ReadVariableOp2@
dense_27/MatMul/ReadVariableOpdense_27/MatMul/ReadVariableOp2?
Btext_vectorization_4/string_lookup_9/None_Lookup/LookupTableFindV2Btext_vectorization_4/string_lookup_9/None_Lookup/LookupTableFindV2:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
*__inference_dense_24_layer_call_fn_2466244

inputs
unknown:	?N
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_24_layer_call_and_return_conditional_losses_2465350o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????N: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????N
 
_user_specified_nameinputs
?
e
,__inference_dropout_11_layer_call_fn_2466312

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dropout_11_layer_call_and_return_conditional_losses_2465489o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
e
G__inference_dropout_10_layer_call_and_return_conditional_losses_2465361

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
/__inference_ffnn_on_count_layer_call_fn_2466000

inputs
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:	?N
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:

	unknown_8:

	unknown_9:


unknown_10:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_ffnn_on_count_layer_call_and_return_conditional_losses_2465422o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?	
f
G__inference_dropout_10_layer_call_and_return_conditional_losses_2465522

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
*__inference_dense_25_layer_call_fn_2466291

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_25_layer_call_and_return_conditional_losses_2465374o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
e
G__inference_dropout_11_layer_call_and_return_conditional_losses_2466317

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
0
 __inference__initializer_2466397
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?

?
E__inference_dense_24_layer_call_and_return_conditional_losses_2465350

inputs1
matmul_readvariableop_resource:	?N-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?N*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????N: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????N
 
_user_specified_nameinputs
?V
?
 __inference__traced_save_2466602
file_prefix.
*savev2_dense_24_kernel_read_readvariableop,
(savev2_dense_24_bias_read_readvariableop.
*savev2_dense_25_kernel_read_readvariableop,
(savev2_dense_25_bias_read_readvariableop.
*savev2_dense_26_kernel_read_readvariableop,
(savev2_dense_26_bias_read_readvariableop.
*savev2_dense_27_kernel_read_readvariableop,
(savev2_dense_27_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableopJ
Fsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2L
Hsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2_1	&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop/
+savev2_true_positives_1_read_readvariableop/
+savev2_true_negatives_1_read_readvariableop0
,savev2_false_positives_1_read_readvariableop0
,savev2_false_negatives_1_read_readvariableop-
)savev2_true_positives_read_readvariableop-
)savev2_true_negatives_read_readvariableop.
*savev2_false_positives_read_readvariableop.
*savev2_false_negatives_read_readvariableop5
1savev2_adam_dense_24_kernel_m_read_readvariableop3
/savev2_adam_dense_24_bias_m_read_readvariableop5
1savev2_adam_dense_25_kernel_m_read_readvariableop3
/savev2_adam_dense_25_bias_m_read_readvariableop5
1savev2_adam_dense_26_kernel_m_read_readvariableop3
/savev2_adam_dense_26_bias_m_read_readvariableop5
1savev2_adam_dense_27_kernel_m_read_readvariableop3
/savev2_adam_dense_27_bias_m_read_readvariableop5
1savev2_adam_dense_24_kernel_v_read_readvariableop3
/savev2_adam_dense_24_bias_v_read_readvariableop5
1savev2_adam_dense_25_kernel_v_read_readvariableop3
/savev2_adam_dense_25_bias_v_read_readvariableop5
1savev2_adam_dense_26_kernel_v_read_readvariableop3
/savev2_adam_dense_26_bias_v_read_readvariableop5
1savev2_adam_dense_27_kernel_v_read_readvariableop3
/savev2_adam_dense_27_bias_v_read_readvariableop
savev2_const_6

identity_1??MergeV2Checkpointsw
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
_temp/part?
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
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:,*
dtype0*?
value?B?,B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEBFlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table-keysBHlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table-valuesB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/3/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/3/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/3/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/3/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:,*
dtype0*k
valuebB`,B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_24_kernel_read_readvariableop(savev2_dense_24_bias_read_readvariableop*savev2_dense_25_kernel_read_readvariableop(savev2_dense_25_bias_read_readvariableop*savev2_dense_26_kernel_read_readvariableop(savev2_dense_26_bias_read_readvariableop*savev2_dense_27_kernel_read_readvariableop(savev2_dense_27_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableopFsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2Hsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2_1"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop+savev2_true_positives_1_read_readvariableop+savev2_true_negatives_1_read_readvariableop,savev2_false_positives_1_read_readvariableop,savev2_false_negatives_1_read_readvariableop)savev2_true_positives_read_readvariableop)savev2_true_negatives_read_readvariableop*savev2_false_positives_read_readvariableop*savev2_false_negatives_read_readvariableop1savev2_adam_dense_24_kernel_m_read_readvariableop/savev2_adam_dense_24_bias_m_read_readvariableop1savev2_adam_dense_25_kernel_m_read_readvariableop/savev2_adam_dense_25_bias_m_read_readvariableop1savev2_adam_dense_26_kernel_m_read_readvariableop/savev2_adam_dense_26_bias_m_read_readvariableop1savev2_adam_dense_27_kernel_m_read_readvariableop/savev2_adam_dense_27_bias_m_read_readvariableop1savev2_adam_dense_24_kernel_v_read_readvariableop/savev2_adam_dense_24_bias_v_read_readvariableop1savev2_adam_dense_25_kernel_v_read_readvariableop/savev2_adam_dense_25_bias_v_read_readvariableop1savev2_adam_dense_26_kernel_v_read_readvariableop/savev2_adam_dense_26_bias_v_read_readvariableop1savev2_adam_dense_27_kernel_v_read_readvariableop/savev2_adam_dense_27_bias_v_read_readvariableopsavev2_const_6"/device:CPU:0*
_output_shapes
 *:
dtypes0
.2,		?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*?
_input_shapes?
?: :	?N::::
:
:
:: : : : : ::: : : : :?:?:?:?:?:?:?:?:	?N::::
:
:
::	?N::::
:
:
:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	?N: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:
: 

_output_shapes
:
:$ 

_output_shapes

:
: 

_output_shapes
::	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:%!

_output_shapes
:	?N: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$  

_output_shapes

:
: !

_output_shapes
:
:$" 

_output_shapes

:
: #

_output_shapes
::%$!

_output_shapes
:	?N: %

_output_shapes
::$& 

_output_shapes

:: '

_output_shapes
::$( 

_output_shapes

:
: )

_output_shapes
:
:$* 

_output_shapes

:
: +

_output_shapes
::,

_output_shapes
: 
?
?
/__inference_ffnn_on_count_layer_call_fn_2465710
input_4
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:	?N
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:

	unknown_8:

	unknown_9:


unknown_10:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_ffnn_on_count_layer_call_and_return_conditional_losses_2465654o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_4:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
.
__inference__destroyer_2466402
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?

?
E__inference_dense_27_layer_call_and_return_conditional_losses_2465415

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?	
f
G__inference_dropout_11_layer_call_and_return_conditional_losses_2465489

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
 __inference__initializer_2466382:
6key_value_init2041753_lookuptableimportv2_table_handle2
.key_value_init2041753_lookuptableimportv2_keys4
0key_value_init2041753_lookuptableimportv2_values	
identity??)key_value_init2041753/LookupTableImportV2?
)key_value_init2041753/LookupTableImportV2LookupTableImportV26key_value_init2041753_lookuptableimportv2_table_handle.key_value_init2041753_lookuptableimportv2_keys0key_value_init2041753_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: r
NoOpNoOp*^key_value_init2041753/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: :?N:?N2V
)key_value_init2041753/LookupTableImportV2)key_value_init2041753/LookupTableImportV2:!

_output_shapes	
:?N:!

_output_shapes	
:?N
?
?
__inference_save_fn_2466421
checkpoint_keyP
Lmutablehashtable_lookup_table_export_values_lookuptableexportv2_table_handle
identity

identity_1

identity_2

identity_3

identity_4

identity_5	???MutableHashTable_lookup_table_export_values/LookupTableExportV2?
?MutableHashTable_lookup_table_export_values/LookupTableExportV2LookupTableExportV2Lmutablehashtable_lookup_table_export_values_lookuptableexportv2_table_handle",/job:localhost/replica:0/task:0/device:CPU:0*
Tkeys0*
Tvalues0	*
_output_shapes

::P
add/yConst*
_output_shapes
: *
dtype0*
valueB B
table-keysK
addAddcheckpoint_keyadd/y:output:0*
T0*
_output_shapes
: T
add_1/yConst*
_output_shapes
: *
dtype0*
valueB Btable-valuesO
add_1Addcheckpoint_keyadd_1/y:output:0*
T0*
_output_shapes
: E
IdentityIdentityadd:z:0^NoOp*
T0*
_output_shapes
: F
ConstConst*
_output_shapes
: *
dtype0*
valueB B N

Identity_1IdentityConst:output:0^NoOp*
T0*
_output_shapes
: ?

Identity_2IdentityFMutableHashTable_lookup_table_export_values/LookupTableExportV2:keys:0^NoOp*
T0*
_output_shapes
:I

Identity_3Identity	add_1:z:0^NoOp*
T0*
_output_shapes
: H
Const_1Const*
_output_shapes
: *
dtype0*
valueB B P

Identity_4IdentityConst_1:output:0^NoOp*
T0*
_output_shapes
: ?

Identity_5IdentityHMutableHashTable_lookup_table_export_values/LookupTableExportV2:values:0^NoOp*
T0	*
_output_shapes
:?
NoOpNoOp@^MutableHashTable_lookup_table_export_values/LookupTableExportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 2?
?MutableHashTable_lookup_table_export_values/LookupTableExportV2?MutableHashTable_lookup_table_export_values/LookupTableExportV2:F B

_output_shapes
: 
(
_user_specified_namecheckpoint_key
?
H
,__inference_dropout_11_layer_call_fn_2466307

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dropout_11_layer_call_and_return_conditional_losses_2465385`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
__inference_<lambda>_2466437:
6key_value_init2041753_lookuptableimportv2_table_handle2
.key_value_init2041753_lookuptableimportv2_keys4
0key_value_init2041753_lookuptableimportv2_values	
identity??)key_value_init2041753/LookupTableImportV2?
)key_value_init2041753/LookupTableImportV2LookupTableImportV26key_value_init2041753_lookuptableimportv2_table_handle.key_value_init2041753_lookuptableimportv2_keys0key_value_init2041753_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: r
NoOpNoOp*^key_value_init2041753/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: :?N:?N2V
)key_value_init2041753/LookupTableImportV2)key_value_init2041753/LookupTableImportV2:!

_output_shapes	
:?N:!

_output_shapes	
:?N
?	
f
G__inference_dropout_10_layer_call_and_return_conditional_losses_2466282

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
E__inference_dense_26_layer_call_and_return_conditional_losses_2466349

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:

identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????
a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
*__inference_dense_26_layer_call_fn_2466338

inputs
unknown:

	unknown_0:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_26_layer_call_and_return_conditional_losses_2465398o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
ӎ
?	
J__inference_ffnn_on_count_layer_call_and_return_conditional_losses_2466125

inputsS
Otext_vectorization_4_string_lookup_9_none_lookup_lookuptablefindv2_table_handleT
Ptext_vectorization_4_string_lookup_9_none_lookup_lookuptablefindv2_default_value	0
,text_vectorization_4_string_lookup_9_equal_y3
/text_vectorization_4_string_lookup_9_selectv2_t	:
'dense_24_matmul_readvariableop_resource:	?N6
(dense_24_biasadd_readvariableop_resource:9
'dense_25_matmul_readvariableop_resource:6
(dense_25_biasadd_readvariableop_resource:9
'dense_26_matmul_readvariableop_resource:
6
(dense_26_biasadd_readvariableop_resource:
9
'dense_27_matmul_readvariableop_resource:
6
(dense_27_biasadd_readvariableop_resource:
identity??dense_24/BiasAdd/ReadVariableOp?dense_24/MatMul/ReadVariableOp?dense_25/BiasAdd/ReadVariableOp?dense_25/MatMul/ReadVariableOp?dense_26/BiasAdd/ReadVariableOp?dense_26/MatMul/ReadVariableOp?dense_27/BiasAdd/ReadVariableOp?dense_27/MatMul/ReadVariableOp?Btext_vectorization_4/string_lookup_9/None_Lookup/LookupTableFindV2`
 text_vectorization_4/StringLowerStringLowerinputs*'
_output_shapes
:??????????
'text_vectorization_4/StaticRegexReplaceStaticRegexReplace)text_vectorization_4/StringLower:output:0*'
_output_shapes
:?????????*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite ?
text_vectorization_4/SqueezeSqueeze0text_vectorization_4/StaticRegexReplace:output:0*
T0*#
_output_shapes
:?????????*
squeeze_dims

?????????g
&text_vectorization_4/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
.text_vectorization_4/StringSplit/StringSplitV2StringSplitV2%text_vectorization_4/Squeeze:output:0/text_vectorization_4/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
4text_vectorization_4/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
6text_vectorization_4/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
6text_vectorization_4/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
.text_vectorization_4/StringSplit/strided_sliceStridedSlice8text_vectorization_4/StringSplit/StringSplitV2:indices:0=text_vectorization_4/StringSplit/strided_slice/stack:output:0?text_vectorization_4/StringSplit/strided_slice/stack_1:output:0?text_vectorization_4/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask?
6text_vectorization_4/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
8text_vectorization_4/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
8text_vectorization_4/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
0text_vectorization_4/StringSplit/strided_slice_1StridedSlice6text_vectorization_4/StringSplit/StringSplitV2:shape:0?text_vectorization_4/StringSplit/strided_slice_1/stack:output:0Atext_vectorization_4/StringSplit/strided_slice_1/stack_1:output:0Atext_vectorization_4/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Wtext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast7text_vectorization_4/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
Ytext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast9text_vectorization_4/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
atext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShape[text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
atext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
`text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdjtext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0jtext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
etext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
ctext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreateritext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ntext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
`text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastgtext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
ctext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
_text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMax[text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0ltext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
atext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
_text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2htext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0jtext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
_text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMuldtext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0ctext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum]text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0ctext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum]text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0gtext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
dtext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincount[text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0gtext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0ltext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
^text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ytext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumktext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0gtext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
btext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
^text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ytext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2ktext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0_text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0gtext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
Btext_vectorization_4/string_lookup_9/None_Lookup/LookupTableFindV2LookupTableFindV2Otext_vectorization_4_string_lookup_9_none_lookup_lookuptablefindv2_table_handle7text_vectorization_4/StringSplit/StringSplitV2:values:0Ptext_vectorization_4_string_lookup_9_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
*text_vectorization_4/string_lookup_9/EqualEqual7text_vectorization_4/StringSplit/StringSplitV2:values:0,text_vectorization_4_string_lookup_9_equal_y*
T0*#
_output_shapes
:??????????
-text_vectorization_4/string_lookup_9/SelectV2SelectV2.text_vectorization_4/string_lookup_9/Equal:z:0/text_vectorization_4_string_lookup_9_selectv2_tKtext_vectorization_4/string_lookup_9/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
-text_vectorization_4/string_lookup_9/IdentityIdentity6text_vectorization_4/string_lookup_9/SelectV2:output:0*
T0	*#
_output_shapes
:??????????
3text_vectorization_4/string_lookup_9/bincount/ShapeShape6text_vectorization_4/string_lookup_9/Identity:output:0*
T0	*
_output_shapes
:}
3text_vectorization_4/string_lookup_9/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
2text_vectorization_4/string_lookup_9/bincount/ProdProd<text_vectorization_4/string_lookup_9/bincount/Shape:output:0<text_vectorization_4/string_lookup_9/bincount/Const:output:0*
T0*
_output_shapes
: y
7text_vectorization_4/string_lookup_9/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
5text_vectorization_4/string_lookup_9/bincount/GreaterGreater;text_vectorization_4/string_lookup_9/bincount/Prod:output:0@text_vectorization_4/string_lookup_9/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
2text_vectorization_4/string_lookup_9/bincount/CastCast9text_vectorization_4/string_lookup_9/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 
5text_vectorization_4/string_lookup_9/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
=text_vectorization_4/string_lookup_9/bincount/RaggedReduceMaxMax6text_vectorization_4/string_lookup_9/Identity:output:0>text_vectorization_4/string_lookup_9/bincount/Const_1:output:0*
T0	*
_output_shapes
: u
3text_vectorization_4/string_lookup_9/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R?
1text_vectorization_4/string_lookup_9/bincount/addAddV2Ftext_vectorization_4/string_lookup_9/bincount/RaggedReduceMax:output:0<text_vectorization_4/string_lookup_9/bincount/add/y:output:0*
T0	*
_output_shapes
: ?
1text_vectorization_4/string_lookup_9/bincount/mulMul6text_vectorization_4/string_lookup_9/bincount/Cast:y:05text_vectorization_4/string_lookup_9/bincount/add:z:0*
T0	*
_output_shapes
: z
7text_vectorization_4/string_lookup_9/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value
B	 R?N?
5text_vectorization_4/string_lookup_9/bincount/MaximumMaximum@text_vectorization_4/string_lookup_9/bincount/minlength:output:05text_vectorization_4/string_lookup_9/bincount/mul:z:0*
T0	*
_output_shapes
: z
7text_vectorization_4/string_lookup_9/bincount/maxlengthConst*
_output_shapes
: *
dtype0	*
value
B	 R?N?
5text_vectorization_4/string_lookup_9/bincount/MinimumMinimum@text_vectorization_4/string_lookup_9/bincount/maxlength:output:09text_vectorization_4/string_lookup_9/bincount/Maximum:z:0*
T0	*
_output_shapes
: x
5text_vectorization_4/string_lookup_9/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB ?
<text_vectorization_4/string_lookup_9/bincount/RaggedBincountRaggedBincountbtext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat:output:06text_vectorization_4/string_lookup_9/Identity:output:09text_vectorization_4/string_lookup_9/bincount/Minimum:z:0>text_vectorization_4/string_lookup_9/bincount/Const_2:output:0*
T0*

Tidx0	*(
_output_shapes
:??????????N?
dense_24/MatMul/ReadVariableOpReadVariableOp'dense_24_matmul_readvariableop_resource*
_output_shapes
:	?N*
dtype0?
dense_24/MatMulMatMulEtext_vectorization_4/string_lookup_9/bincount/RaggedBincount:output:0&dense_24/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_24/BiasAdd/ReadVariableOpReadVariableOp(dense_24_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_24/BiasAddBiasAdddense_24/MatMul:product:0'dense_24/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????b
dense_24/ReluReludense_24/BiasAdd:output:0*
T0*'
_output_shapes
:?????????n
dropout_10/IdentityIdentitydense_24/Relu:activations:0*
T0*'
_output_shapes
:??????????
dense_25/MatMul/ReadVariableOpReadVariableOp'dense_25_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
dense_25/MatMulMatMuldropout_10/Identity:output:0&dense_25/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_25/BiasAdd/ReadVariableOpReadVariableOp(dense_25_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_25/BiasAddBiasAdddense_25/MatMul:product:0'dense_25/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????b
dense_25/ReluReludense_25/BiasAdd:output:0*
T0*'
_output_shapes
:?????????n
dropout_11/IdentityIdentitydense_25/Relu:activations:0*
T0*'
_output_shapes
:??????????
dense_26/MatMul/ReadVariableOpReadVariableOp'dense_26_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0?
dense_26/MatMulMatMuldropout_11/Identity:output:0&dense_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
dense_26/BiasAdd/ReadVariableOpReadVariableOp(dense_26_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0?
dense_26/BiasAddBiasAdddense_26/MatMul:product:0'dense_26/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
b
dense_26/ReluReludense_26/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
?
dense_27/MatMul/ReadVariableOpReadVariableOp'dense_27_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0?
dense_27/MatMulMatMuldense_26/Relu:activations:0&dense_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_27/BiasAdd/ReadVariableOpReadVariableOp(dense_27_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_27/BiasAddBiasAdddense_27/MatMul:product:0'dense_27/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????h
dense_27/SigmoidSigmoiddense_27/BiasAdd:output:0*
T0*'
_output_shapes
:?????????c
IdentityIdentitydense_27/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_24/BiasAdd/ReadVariableOp^dense_24/MatMul/ReadVariableOp ^dense_25/BiasAdd/ReadVariableOp^dense_25/MatMul/ReadVariableOp ^dense_26/BiasAdd/ReadVariableOp^dense_26/MatMul/ReadVariableOp ^dense_27/BiasAdd/ReadVariableOp^dense_27/MatMul/ReadVariableOpC^text_vectorization_4/string_lookup_9/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????: : : : : : : : : : : : 2B
dense_24/BiasAdd/ReadVariableOpdense_24/BiasAdd/ReadVariableOp2@
dense_24/MatMul/ReadVariableOpdense_24/MatMul/ReadVariableOp2B
dense_25/BiasAdd/ReadVariableOpdense_25/BiasAdd/ReadVariableOp2@
dense_25/MatMul/ReadVariableOpdense_25/MatMul/ReadVariableOp2B
dense_26/BiasAdd/ReadVariableOpdense_26/BiasAdd/ReadVariableOp2@
dense_26/MatMul/ReadVariableOpdense_26/MatMul/ReadVariableOp2B
dense_27/BiasAdd/ReadVariableOpdense_27/BiasAdd/ReadVariableOp2@
dense_27/MatMul/ReadVariableOpdense_27/MatMul/ReadVariableOp2?
Btext_vectorization_4/string_lookup_9/None_Lookup/LookupTableFindV2Btext_vectorization_4/string_lookup_9/None_Lookup/LookupTableFindV2:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
??
?
J__inference_ffnn_on_count_layer_call_and_return_conditional_losses_2465798
input_4S
Otext_vectorization_4_string_lookup_9_none_lookup_lookuptablefindv2_table_handleT
Ptext_vectorization_4_string_lookup_9_none_lookup_lookuptablefindv2_default_value	0
,text_vectorization_4_string_lookup_9_equal_y3
/text_vectorization_4_string_lookup_9_selectv2_t	#
dense_24_2465775:	?N
dense_24_2465777:"
dense_25_2465781:
dense_25_2465783:"
dense_26_2465787:

dense_26_2465789:
"
dense_27_2465792:

dense_27_2465794:
identity?? dense_24/StatefulPartitionedCall? dense_25/StatefulPartitionedCall? dense_26/StatefulPartitionedCall? dense_27/StatefulPartitionedCall?Btext_vectorization_4/string_lookup_9/None_Lookup/LookupTableFindV2a
 text_vectorization_4/StringLowerStringLowerinput_4*'
_output_shapes
:??????????
'text_vectorization_4/StaticRegexReplaceStaticRegexReplace)text_vectorization_4/StringLower:output:0*'
_output_shapes
:?????????*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite ?
text_vectorization_4/SqueezeSqueeze0text_vectorization_4/StaticRegexReplace:output:0*
T0*#
_output_shapes
:?????????*
squeeze_dims

?????????g
&text_vectorization_4/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
.text_vectorization_4/StringSplit/StringSplitV2StringSplitV2%text_vectorization_4/Squeeze:output:0/text_vectorization_4/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
4text_vectorization_4/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
6text_vectorization_4/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
6text_vectorization_4/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
.text_vectorization_4/StringSplit/strided_sliceStridedSlice8text_vectorization_4/StringSplit/StringSplitV2:indices:0=text_vectorization_4/StringSplit/strided_slice/stack:output:0?text_vectorization_4/StringSplit/strided_slice/stack_1:output:0?text_vectorization_4/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask?
6text_vectorization_4/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
8text_vectorization_4/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
8text_vectorization_4/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
0text_vectorization_4/StringSplit/strided_slice_1StridedSlice6text_vectorization_4/StringSplit/StringSplitV2:shape:0?text_vectorization_4/StringSplit/strided_slice_1/stack:output:0Atext_vectorization_4/StringSplit/strided_slice_1/stack_1:output:0Atext_vectorization_4/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Wtext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast7text_vectorization_4/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
Ytext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast9text_vectorization_4/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
atext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShape[text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
atext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
`text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdjtext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0jtext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
etext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
ctext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreateritext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ntext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
`text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastgtext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
ctext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
_text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMax[text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0ltext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
atext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
_text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2htext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0jtext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
_text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMuldtext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0ctext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum]text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0ctext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum]text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0gtext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
dtext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincount[text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0gtext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0ltext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
^text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ytext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumktext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0gtext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
btext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
^text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ytext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2ktext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0_text_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0gtext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
Btext_vectorization_4/string_lookup_9/None_Lookup/LookupTableFindV2LookupTableFindV2Otext_vectorization_4_string_lookup_9_none_lookup_lookuptablefindv2_table_handle7text_vectorization_4/StringSplit/StringSplitV2:values:0Ptext_vectorization_4_string_lookup_9_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
*text_vectorization_4/string_lookup_9/EqualEqual7text_vectorization_4/StringSplit/StringSplitV2:values:0,text_vectorization_4_string_lookup_9_equal_y*
T0*#
_output_shapes
:??????????
-text_vectorization_4/string_lookup_9/SelectV2SelectV2.text_vectorization_4/string_lookup_9/Equal:z:0/text_vectorization_4_string_lookup_9_selectv2_tKtext_vectorization_4/string_lookup_9/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
-text_vectorization_4/string_lookup_9/IdentityIdentity6text_vectorization_4/string_lookup_9/SelectV2:output:0*
T0	*#
_output_shapes
:??????????
3text_vectorization_4/string_lookup_9/bincount/ShapeShape6text_vectorization_4/string_lookup_9/Identity:output:0*
T0	*
_output_shapes
:}
3text_vectorization_4/string_lookup_9/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
2text_vectorization_4/string_lookup_9/bincount/ProdProd<text_vectorization_4/string_lookup_9/bincount/Shape:output:0<text_vectorization_4/string_lookup_9/bincount/Const:output:0*
T0*
_output_shapes
: y
7text_vectorization_4/string_lookup_9/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
5text_vectorization_4/string_lookup_9/bincount/GreaterGreater;text_vectorization_4/string_lookup_9/bincount/Prod:output:0@text_vectorization_4/string_lookup_9/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
2text_vectorization_4/string_lookup_9/bincount/CastCast9text_vectorization_4/string_lookup_9/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 
5text_vectorization_4/string_lookup_9/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
=text_vectorization_4/string_lookup_9/bincount/RaggedReduceMaxMax6text_vectorization_4/string_lookup_9/Identity:output:0>text_vectorization_4/string_lookup_9/bincount/Const_1:output:0*
T0	*
_output_shapes
: u
3text_vectorization_4/string_lookup_9/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R?
1text_vectorization_4/string_lookup_9/bincount/addAddV2Ftext_vectorization_4/string_lookup_9/bincount/RaggedReduceMax:output:0<text_vectorization_4/string_lookup_9/bincount/add/y:output:0*
T0	*
_output_shapes
: ?
1text_vectorization_4/string_lookup_9/bincount/mulMul6text_vectorization_4/string_lookup_9/bincount/Cast:y:05text_vectorization_4/string_lookup_9/bincount/add:z:0*
T0	*
_output_shapes
: z
7text_vectorization_4/string_lookup_9/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value
B	 R?N?
5text_vectorization_4/string_lookup_9/bincount/MaximumMaximum@text_vectorization_4/string_lookup_9/bincount/minlength:output:05text_vectorization_4/string_lookup_9/bincount/mul:z:0*
T0	*
_output_shapes
: z
7text_vectorization_4/string_lookup_9/bincount/maxlengthConst*
_output_shapes
: *
dtype0	*
value
B	 R?N?
5text_vectorization_4/string_lookup_9/bincount/MinimumMinimum@text_vectorization_4/string_lookup_9/bincount/maxlength:output:09text_vectorization_4/string_lookup_9/bincount/Maximum:z:0*
T0	*
_output_shapes
: x
5text_vectorization_4/string_lookup_9/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB ?
<text_vectorization_4/string_lookup_9/bincount/RaggedBincountRaggedBincountbtext_vectorization_4/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat:output:06text_vectorization_4/string_lookup_9/Identity:output:09text_vectorization_4/string_lookup_9/bincount/Minimum:z:0>text_vectorization_4/string_lookup_9/bincount/Const_2:output:0*
T0*

Tidx0	*(
_output_shapes
:??????????N?
 dense_24/StatefulPartitionedCallStatefulPartitionedCallEtext_vectorization_4/string_lookup_9/bincount/RaggedBincount:output:0dense_24_2465775dense_24_2465777*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_24_layer_call_and_return_conditional_losses_2465350?
dropout_10/PartitionedCallPartitionedCall)dense_24/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dropout_10_layer_call_and_return_conditional_losses_2465361?
 dense_25/StatefulPartitionedCallStatefulPartitionedCall#dropout_10/PartitionedCall:output:0dense_25_2465781dense_25_2465783*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_25_layer_call_and_return_conditional_losses_2465374?
dropout_11/PartitionedCallPartitionedCall)dense_25/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dropout_11_layer_call_and_return_conditional_losses_2465385?
 dense_26/StatefulPartitionedCallStatefulPartitionedCall#dropout_11/PartitionedCall:output:0dense_26_2465787dense_26_2465789*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_26_layer_call_and_return_conditional_losses_2465398?
 dense_27/StatefulPartitionedCallStatefulPartitionedCall)dense_26/StatefulPartitionedCall:output:0dense_27_2465792dense_27_2465794*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_27_layer_call_and_return_conditional_losses_2465415x
IdentityIdentity)dense_27/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp!^dense_24/StatefulPartitionedCall!^dense_25/StatefulPartitionedCall!^dense_26/StatefulPartitionedCall!^dense_27/StatefulPartitionedCallC^text_vectorization_4/string_lookup_9/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????: : : : : : : : : : : : 2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall2D
 dense_25/StatefulPartitionedCall dense_25/StatefulPartitionedCall2D
 dense_26/StatefulPartitionedCall dense_26/StatefulPartitionedCall2D
 dense_27/StatefulPartitionedCall dense_27/StatefulPartitionedCall2?
Btext_vectorization_4/string_lookup_9/None_Lookup/LookupTableFindV2Btext_vectorization_4/string_lookup_9/None_Lookup/LookupTableFindV2:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_4:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
__inference_restore_fn_2466429
restored_tensors_0
restored_tensors_1	C
?mutablehashtable_table_restore_lookuptableimportv2_table_handle
identity??2MutableHashTable_table_restore/LookupTableImportV2?
2MutableHashTable_table_restore/LookupTableImportV2LookupTableImportV2?mutablehashtable_table_restore_lookuptableimportv2_table_handlerestored_tensors_0restored_tensors_1",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: {
NoOpNoOp3^MutableHashTable_table_restore/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes

::: 2h
2MutableHashTable_table_restore/LookupTableImportV22MutableHashTable_table_restore/LookupTableImportV2:L H

_output_shapes
:
,
_user_specified_namerestored_tensors_0:LH

_output_shapes
:
,
_user_specified_namerestored_tensors_1
?
e
,__inference_dropout_10_layer_call_fn_2466265

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dropout_10_layer_call_and_return_conditional_losses_2465522o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
E__inference_dense_27_layer_call_and_return_conditional_losses_2466369

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_2:0StatefulPartitionedCall_38"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
;
input_40
serving_default_input_4:0?????????>
dense_272
StatefulPartitionedCall_1:0?????????tensorflow/serving/predict:??
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
	variables
	trainable_variables

regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
P
	keras_api
_lookup_layer
_adapt_function"
_tf_keras_layer
?
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
?
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses
"_random_generator"
_tf_keras_layer
?
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses

)kernel
*bias"
_tf_keras_layer
?
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/__call__
*0&call_and_return_all_conditional_losses
1_random_generator"
_tf_keras_layer
?
2	variables
3trainable_variables
4regularization_losses
5	keras_api
6__call__
*7&call_and_return_all_conditional_losses

8kernel
9bias"
_tf_keras_layer
?
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses

@kernel
Abias"
_tf_keras_layer
X
1
2
)3
*4
85
96
@7
A8"
trackable_list_wrapper
X
0
1
)2
*3
84
95
@6
A7"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Bnon_trainable_variables

Clayers
Dmetrics
Elayer_regularization_losses
Flayer_metrics
	variables
	trainable_variables

regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?
Gtrace_0
Htrace_1
Itrace_2
Jtrace_32?
/__inference_ffnn_on_count_layer_call_fn_2465449
/__inference_ffnn_on_count_layer_call_fn_2466000
/__inference_ffnn_on_count_layer_call_fn_2466029
/__inference_ffnn_on_count_layer_call_fn_2465710?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 zGtrace_0zHtrace_1zItrace_2zJtrace_3
?
Ktrace_0
Ltrace_1
Mtrace_2
Ntrace_32?
J__inference_ffnn_on_count_layer_call_and_return_conditional_losses_2466125
J__inference_ffnn_on_count_layer_call_and_return_conditional_losses_2466235
J__inference_ffnn_on_count_layer_call_and_return_conditional_losses_2465798
J__inference_ffnn_on_count_layer_call_and_return_conditional_losses_2465886?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 zKtrace_0zLtrace_1zMtrace_2zNtrace_3
?B?
"__inference__wrapped_model_2465270input_4"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?
Oiter

Pbeta_1

Qbeta_2
	Rdecay
Slearning_ratem?m?)m?*m?8m?9m?@m?Am?v?v?)v?*v?8v?9v?@v?Av?"
	optimizer
,
Tserving_default"
signature_map
"
_generic_user_object
L
U	keras_api
Vlookup_table
Wtoken_counts"
_tf_keras_layer
?
Xtrace_02?
__inference_adapt_step_2465971?
???
FullArgSpec
args?

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 zXtrace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Ynon_trainable_variables

Zlayers
[metrics
\layer_regularization_losses
]layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?
^trace_02?
*__inference_dense_24_layer_call_fn_2466244?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z^trace_0
?
_trace_02?
E__inference_dense_24_layer_call_and_return_conditional_losses_2466255?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z_trace_0
": 	?N2dense_24/kernel
:2dense_24/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
`non_trainable_variables

alayers
bmetrics
clayer_regularization_losses
dlayer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses"
_generic_user_object
?
etrace_0
ftrace_12?
,__inference_dropout_10_layer_call_fn_2466260
,__inference_dropout_10_layer_call_fn_2466265?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 zetrace_0zftrace_1
?
gtrace_0
htrace_12?
G__inference_dropout_10_layer_call_and_return_conditional_losses_2466270
G__inference_dropout_10_layer_call_and_return_conditional_losses_2466282?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 zgtrace_0zhtrace_1
"
_generic_user_object
.
)0
*1"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
inon_trainable_variables

jlayers
kmetrics
llayer_regularization_losses
mlayer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses"
_generic_user_object
?
ntrace_02?
*__inference_dense_25_layer_call_fn_2466291?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 zntrace_0
?
otrace_02?
E__inference_dense_25_layer_call_and_return_conditional_losses_2466302?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 zotrace_0
!:2dense_25/kernel
:2dense_25/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
pnon_trainable_variables

qlayers
rmetrics
slayer_regularization_losses
tlayer_metrics
+	variables
,trainable_variables
-regularization_losses
/__call__
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses"
_generic_user_object
?
utrace_0
vtrace_12?
,__inference_dropout_11_layer_call_fn_2466307
,__inference_dropout_11_layer_call_fn_2466312?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 zutrace_0zvtrace_1
?
wtrace_0
xtrace_12?
G__inference_dropout_11_layer_call_and_return_conditional_losses_2466317
G__inference_dropout_11_layer_call_and_return_conditional_losses_2466329?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 zwtrace_0zxtrace_1
"
_generic_user_object
.
80
91"
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
 "
trackable_list_wrapper
?
ynon_trainable_variables

zlayers
{metrics
|layer_regularization_losses
}layer_metrics
2	variables
3trainable_variables
4regularization_losses
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses"
_generic_user_object
?
~trace_02?
*__inference_dense_26_layer_call_fn_2466338?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z~trace_0
?
trace_02?
E__inference_dense_26_layer_call_and_return_conditional_losses_2466349?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ztrace_0
!:
2dense_26/kernel
:
2dense_26/bias
.
@0
A1"
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
*__inference_dense_27_layer_call_fn_2466358?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
E__inference_dense_27_layer_call_and_return_conditional_losses_2466369?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
!:
2dense_27/kernel
:2dense_27/bias
 "
trackable_list_wrapper
Q
0
1
2
3
4
5
6"
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
/__inference_ffnn_on_count_layer_call_fn_2465449input_4"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
/__inference_ffnn_on_count_layer_call_fn_2466000inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
/__inference_ffnn_on_count_layer_call_fn_2466029inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
/__inference_ffnn_on_count_layer_call_fn_2465710input_4"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
J__inference_ffnn_on_count_layer_call_and_return_conditional_losses_2466125inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
J__inference_ffnn_on_count_layer_call_and_return_conditional_losses_2466235inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
J__inference_ffnn_on_count_layer_call_and_return_conditional_losses_2465798input_4"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
J__inference_ffnn_on_count_layer_call_and_return_conditional_losses_2465886input_4"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
?B?
%__inference_signature_wrapper_2465923input_4"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
"
_generic_user_object
j
?_initializer
?_create_resource
?_initialize
?_destroy_resourceR jtf.StaticHashTable
O
?_create_resource
?_initialize
?_destroy_resourceR Z

 ??
?B?
__inference_adapt_step_2465971iterator"?
???
FullArgSpec
args?

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
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
?B?
*__inference_dense_24_layer_call_fn_2466244inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
E__inference_dense_24_layer_call_and_return_conditional_losses_2466255inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
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
?B?
,__inference_dropout_10_layer_call_fn_2466260inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
,__inference_dropout_10_layer_call_fn_2466265inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
G__inference_dropout_10_layer_call_and_return_conditional_losses_2466270inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
G__inference_dropout_10_layer_call_and_return_conditional_losses_2466282inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
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
?B?
*__inference_dense_25_layer_call_fn_2466291inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
E__inference_dense_25_layer_call_and_return_conditional_losses_2466302inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
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
?B?
,__inference_dropout_11_layer_call_fn_2466307inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
,__inference_dropout_11_layer_call_fn_2466312inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
G__inference_dropout_11_layer_call_and_return_conditional_losses_2466317inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
G__inference_dropout_11_layer_call_and_return_conditional_losses_2466329inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
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
?B?
*__inference_dense_26_layer_call_fn_2466338inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
E__inference_dense_26_layer_call_and_return_conditional_losses_2466349inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
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
?B?
*__inference_dense_27_layer_call_fn_2466358inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
E__inference_dense_27_layer_call_and_return_conditional_losses_2466369inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
R
?	variables
?	keras_api

?total

?count"
_tf_keras_metric
c
?	variables
?	keras_api

?total

?count
?
_fn_kwargs"
_tf_keras_metric
?
?	variables
?	keras_api
?true_positives
?true_negatives
?false_positives
?false_negatives"
_tf_keras_metric
?
?	variables
?	keras_api
?true_positives
?true_negatives
?false_positives
?false_negatives"
_tf_keras_metric
"
_generic_user_object
?
?trace_02?
__inference__creator_2466374?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z?trace_0
?
?trace_02?
 __inference__initializer_2466382?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z?trace_0
?
?trace_02?
__inference__destroyer_2466387?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z?trace_0
?
?trace_02?
__inference__creator_2466392?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z?trace_0
?
?trace_02?
 __inference__initializer_2466397?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z?trace_0
?
?trace_02?
__inference__destroyer_2466402?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z?trace_0
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:? (2true_positives
:? (2true_negatives
 :? (2false_positives
 :? (2false_negatives
@
?0
?1
?2
?3"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:? (2true_positives
:? (2true_negatives
 :? (2false_positives
 :? (2false_negatives
?B?
__inference__creator_2466374"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
 __inference__initializer_2466382"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
__inference__destroyer_2466387"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
__inference__creator_2466392"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
 __inference__initializer_2466397"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
__inference__destroyer_2466402"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
':%	?N2Adam/dense_24/kernel/m
 :2Adam/dense_24/bias/m
&:$2Adam/dense_25/kernel/m
 :2Adam/dense_25/bias/m
&:$
2Adam/dense_26/kernel/m
 :
2Adam/dense_26/bias/m
&:$
2Adam/dense_27/kernel/m
 :2Adam/dense_27/bias/m
':%	?N2Adam/dense_24/kernel/v
 :2Adam/dense_24/bias/v
&:$2Adam/dense_25/kernel/v
 :2Adam/dense_25/bias/v
&:$
2Adam/dense_26/kernel/v
 :
2Adam/dense_26/bias/v
&:$
2Adam/dense_27/kernel/v
 :2Adam/dense_27/bias/v
?B?
__inference_save_fn_2466421checkpoint_key"?
???
FullArgSpec
args?
jcheckpoint_key
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?	
? 
?B?
__inference_restore_fn_2466429restored_tensors_0restored_tensors_1"?
???
FullArgSpec
args? 
varargsjrestored_tensors
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?
	?
	?	
J
Constjtf.TrackableConstant
!J	
Const_1jtf.TrackableConstant
!J	
Const_2jtf.TrackableConstant
!J	
Const_3jtf.TrackableConstant
!J	
Const_4jtf.TrackableConstant
!J	
Const_5jtf.TrackableConstant8
__inference__creator_2466374?

? 
? "? 8
__inference__creator_2466392?

? 
? "? :
__inference__destroyer_2466387?

? 
? "? :
__inference__destroyer_2466402?

? 
? "? C
 __inference__initializer_2466382V???

? 
? "? <
 __inference__initializer_2466397?

? 
? "? ?
"__inference__wrapped_model_2465270xV???)*89@A0?-
&?#
!?
input_4?????????
? "3?0
.
dense_27"?
dense_27?????????h
__inference_adapt_step_2465971FW?;?8
1?.
,?)?
?	?IteratorSpec 
? "
 ?
E__inference_dense_24_layer_call_and_return_conditional_losses_2466255]0?-
&?#
!?
inputs??????????N
? "%?"
?
0?????????
? ~
*__inference_dense_24_layer_call_fn_2466244P0?-
&?#
!?
inputs??????????N
? "???????????
E__inference_dense_25_layer_call_and_return_conditional_losses_2466302\)*/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? }
*__inference_dense_25_layer_call_fn_2466291O)*/?,
%?"
 ?
inputs?????????
? "???????????
E__inference_dense_26_layer_call_and_return_conditional_losses_2466349\89/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????

? }
*__inference_dense_26_layer_call_fn_2466338O89/?,
%?"
 ?
inputs?????????
? "??????????
?
E__inference_dense_27_layer_call_and_return_conditional_losses_2466369\@A/?,
%?"
 ?
inputs?????????

? "%?"
?
0?????????
? }
*__inference_dense_27_layer_call_fn_2466358O@A/?,
%?"
 ?
inputs?????????

? "???????????
G__inference_dropout_10_layer_call_and_return_conditional_losses_2466270\3?0
)?&
 ?
inputs?????????
p 
? "%?"
?
0?????????
? ?
G__inference_dropout_10_layer_call_and_return_conditional_losses_2466282\3?0
)?&
 ?
inputs?????????
p
? "%?"
?
0?????????
? 
,__inference_dropout_10_layer_call_fn_2466260O3?0
)?&
 ?
inputs?????????
p 
? "??????????
,__inference_dropout_10_layer_call_fn_2466265O3?0
)?&
 ?
inputs?????????
p
? "???????????
G__inference_dropout_11_layer_call_and_return_conditional_losses_2466317\3?0
)?&
 ?
inputs?????????
p 
? "%?"
?
0?????????
? ?
G__inference_dropout_11_layer_call_and_return_conditional_losses_2466329\3?0
)?&
 ?
inputs?????????
p
? "%?"
?
0?????????
? 
,__inference_dropout_11_layer_call_fn_2466307O3?0
)?&
 ?
inputs?????????
p 
? "??????????
,__inference_dropout_11_layer_call_fn_2466312O3?0
)?&
 ?
inputs?????????
p
? "???????????
J__inference_ffnn_on_count_layer_call_and_return_conditional_losses_2465798rV???)*89@A8?5
.?+
!?
input_4?????????
p 

 
? "%?"
?
0?????????
? ?
J__inference_ffnn_on_count_layer_call_and_return_conditional_losses_2465886rV???)*89@A8?5
.?+
!?
input_4?????????
p

 
? "%?"
?
0?????????
? ?
J__inference_ffnn_on_count_layer_call_and_return_conditional_losses_2466125qV???)*89@A7?4
-?*
 ?
inputs?????????
p 

 
? "%?"
?
0?????????
? ?
J__inference_ffnn_on_count_layer_call_and_return_conditional_losses_2466235qV???)*89@A7?4
-?*
 ?
inputs?????????
p

 
? "%?"
?
0?????????
? ?
/__inference_ffnn_on_count_layer_call_fn_2465449eV???)*89@A8?5
.?+
!?
input_4?????????
p 

 
? "???????????
/__inference_ffnn_on_count_layer_call_fn_2465710eV???)*89@A8?5
.?+
!?
input_4?????????
p

 
? "???????????
/__inference_ffnn_on_count_layer_call_fn_2466000dV???)*89@A7?4
-?*
 ?
inputs?????????
p 

 
? "???????????
/__inference_ffnn_on_count_layer_call_fn_2466029dV???)*89@A7?4
-?*
 ?
inputs?????????
p

 
? "??????????{
__inference_restore_fn_2466429YWK?H
A?>
?
restored_tensors_0
?
restored_tensors_1	
? "? ?
__inference_save_fn_2466421?W&?#
?
?
checkpoint_key 
? "???
`?]

name?
0/name 
#

slice_spec?
0/slice_spec 

tensor?
0/tensor
`?]

name?
1/name 
#

slice_spec?
1/slice_spec 

tensor?
1/tensor	?
%__inference_signature_wrapper_2465923?V???)*89@A;?8
? 
1?.
,
input_4!?
input_4?????????"3?0
.
dense_27"?
dense_27?????????