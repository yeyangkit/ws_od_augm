import numpy as np
import os



import matplotlib.pyplot as plt

ap_baseline = [[0.951323054463045, 0.624837031971126, 0.7989415131081208, 0.560225072844051, 0.011181939783966298, 0.3771804938788751, 0.6625557642974114, 0.0, 0.0, 0.40887210162807436], [0.8639392970749108, 0.4555836819962774, 0.4854632980718092, 0.06748060904028347, 0.049410760800349625, 0.3199763423258453, 0.30335315176763294, 0.013263209794662278, 0.0, 0.23648672163989892], [0.7836482436744846, 0.2547222697524554, 0.5305826805837106, 0.00811387360357698, 0.19622210510937693, 0.21205640111945584, 0.11374317555167683, 0.02085098507858701, 0.0, 0.1604376507146078], [0.6518991572889983, 0.2501233082982669, 0.2786060951556718, 0.059952932960249866, 0.010656552875349161, 0.12595686909549236, 0.015789073503248117, 0.0, 0.0, 0.06512065478642785], [0.5262736910770067, 0.1353245275611219, 0.1925784961324089, 0.07515986689554652, 0.0, 0.0476601161002292, 0.0, 0.0, 0.0, 0.014732747447541935], [0.40177446770360636, 0.13828684965556054, 0.10277458912537227, 0.024148255938530105, 0.0, 0.012655002778816518, 0.0, 0.0, 0.0, 0.0], [0.22575337595874034, 0.10006245778759376, 0.14156646470855072, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.1518896926870087, 0.031826948821449985, 0.04280346939660552, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.095942657904031, 0.021094462709707155, 0.009800693758807124, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
ap_se = [[0.9187258411358361, 0.4946870265363914, 0.7982876599533841, 0.39731779698535474, 0.0, 0.44084336085974307, 0.6897649862090515, 0.03333962522638512, 0.0, 0.2916036300680781], [0.8435787500860282, 0.361401536030342, 0.5288514162342809, 0.22831987441274354, 0.014432566173744481, 0.404816134500283, 0.3821785323170483, 0.09340109663664353, 0.0, 0.20981420161574957], [0.7706901732519921, 0.27835239833662523, 0.5599384442554505, 0.24833358153739607, 0.008708682546003698, 0.26143076144263844, 0.19714871871077916, 0.03126321099249482, 0.0, 0.21979583480933906], [0.6322012887873939, 0.19195252714506092, 0.3222565033619401, 0.16656311005804414, 0.0073052227762763836, 0.18473348936017897, 0.14074818168486386, 0.009750554181216918, 0.0, 0.0778214989545595], [0.5079916911437506, 0.09326706184232875, 0.20277103577300531, 0.16486867738010388, 0.0, 0.11916616729722672, 0.04005997500335812, 0.0, 0.0, 0.033297101404419466], [0.3672004034994172, 0.11979103886501977, 0.1498900238886675, 0.13630508813703412, 0.0, 0.06971037502882994, 0.00257328986038707, 0.0, 0.0, 0.0], [0.22430147168262274, 0.04920440324688881, 0.15306053513348006, 0.02970492871608009, 0.0, 0.016380827218555306, 0.0004662309368191721, 0.0, 0.0, 0.0], [0.14563699498196378, 0.02173712495521046, 0.04393951526030354, 0.00818211105168971, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.08025749783258115, 0.010178163120638434, 0.0, 0.0037636792005806517, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
ap_seq12 = [[0.9422867036689054, 0.4526288518204511, 0.7932022127810429, 0.5143673308039394, 0.007096132262798928, 0.37185201701463, 0.6101608372970078, 0.03898957160192492, 0.0, 0.348427475601555], [0.8573728603691063, 0.3227297433261611, 0.5590638944431925, 0.1986516573075227, 0.022568268270617795, 0.31668166095300376, 0.26419957631731195, 0.06651477372936494, 0.0, 0.22339734459175914], [0.7626594543529057, 0.21656665649071927, 0.5320645896557976, 0.11117243084491907, 0.20299531327496556, 0.19243716535460442, 0.08186059762516934, 0.03996970895832691, 0.0, 0.28611488660619017], [0.6268458008154884, 0.1441955591418276, 0.3412503931132501, 0.17545968067735224, 0.021058096062690164, 0.12408226261861212, 0.013993193611152532, 0.017549595685274325, 0.0, 0.1259189946686716], [0.48310802969144906, 0.09788309916166574, 0.24828163561868746, 0.14221841656879639, 0.0, 0.06432649840988788, 0.0, 0.0, 0.0, 0.06833521501099944], [0.36287642838196615, 0.06576757784249251, 0.11378102018839359, 0.08739864368681302, 0.0, 0.015026196096312772, 0.0, 0.0, 0.0, 0.0], [0.20293875972261244, 0.036184802098718265, 0.1652380856312186, 0.043779900203188865, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.1416980072534521, 0.00729640927073502, 0.05135136363920664, 0.018527591570725133, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0794999005128163, 0.004295849814722745, 0.00034131859131859133, 0.0031935111254670786, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
ap_seq32 = [[0.9296327305166601, 0.5208438507370918, 0.8624276655638022, 0.42292265572230314, 0.01330942958273419, 0.35909693699715783, 0.7101526073950422, 0.030487524889420345, 0.0, 0.29875170358287145], [0.836499332711548, 0.33598434723516374, 0.5319666831776286, 0.20983012961962727, 0.02555596464804394, 0.3258863671310449, 0.3026299003079375, 0.06178815014237634, 0.0, 0.21743023580564805], [0.7592518932697072, 0.22282482007091353, 0.47694230620730027, 0.1804619062241138, 0.1794762690673765, 0.22023182103854572, 0.10546330544013223, 0.04500382772158061, 0.0, 0.14880454137145663], [0.6610600313519843, 0.16701261203536766, 0.31611678524447095, 0.21815860506820345, 0.012452679827768348, 0.15830791087476812, 0.11234365574623514, 0.008103882259306755, 0.0, 0.0825959394371087], [0.5125912759399486, 0.1260306138072083, 0.2155443392609636, 0.16280798012165507, 0.0, 0.09704198674837254, 0.047828629204776325, 0.0, 0.0, 0.042969909140343876], [0.38994892211238025, 0.10676243983200628, 0.1320859141297875, 0.06667337067505341, 0.0, 0.05839533410256951, 0.0, 0.0, 0.0, 0.0], [0.23871905842352217, 0.08087438458601445, 0.20680455708517542, 0.0190666311102402, 0.0, 0.004785059409310798, 0.0, 0.0, 0.0, 0.0], [0.16236323186207566, 0.030100891499414757, 0.05149097495940591, 0.013951191938568714, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.08656283470625904, 0.0245440614923165, 0.002158202555273698, 0.004138431635910967, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
ap_target = [[0.8969059299606466, 0.41158821137994966, 0.7310613750530346, 0.17237889720720728, 0.07126740888094424, 0.4367543019494005, 0.6248729036710344, 0.06067925765696183, 0.0, 0.2986101859142324], [0.8245744527095054, 0.46677502226636614, 0.43269349659731604, 0.13058932750443872, 0.08335962047500581, 0.4508538785688419, 0.36755933779036154, 0.08276279286871113, 0.0, 0.23048031115802936], [0.8041560487000641, 0.33610734707213963, 0.36816571216673555, 0.12712542098698076, 0.27881325126432077, 0.3306488353820217, 0.17108650819350774, 0.101793801172155, 0.0, 0.1605587866002834], [0.7042714399916842, 0.415777483782229, 0.1751791317912399, 0.1736870866099307, 0.10436458870724025, 0.34147565543506936, 0.10393280207244163, 0.013480749607893026, 0.0, 0.1316429313024584], [0.6132612138979449, 0.26092331511071154, 0.28465228744468724, 0.11152700342506602, 0.011181736514535509, 0.2591377383454039, 0.06687702104355449, 0.0, 0.0, 0.11736709313921741], [0.5461790404250972, 0.20951261772180185, 0.16857761210481462, 0.06712321383423678, 0.031395080057203914, 0.2058248288838975, 0.0, 0.0, 0.0, 0.0], [0.36825893674922305, 0.17671984827945345, 0.1883239519022074, 0.05912255803916168, 0.0003332051553672031, 0.1389061887247739, 0.0, 0.0, 0.0, 0.0], [0.31664757416236655, 0.11693463440855105, 0.0858429275084948, 0.03796053228202809, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.21331088301004114, 0.10720482801976719, 0.1229640429933111, 0.002655003329356928, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
ap_se_extra = [[0.9079685437375906, 0.5595312695502846, 0.6311691733889011, 0.3933605162527529, 0.0008933910168478068, 0.30205674856036663, 0.6101535689424475, 0.0, 0.0, 0.33584072001767357], [0.8359977498818026, 0.3690512895167392, 0.5341499299922473, 0.05508052199797821, 0.05233469295118851, 0.2889883564260436, 0.29939159879640553, 0.03823693401922446, 0.0, 0.23772598590714789], [0.7454050845277853, 0.30398149607976543, 0.4581868017392057, 0.14651689128467132, 0.15098364976922213, 0.16953380690977948, 0.10022181086741085, 0.008770865642668676, 0.0, 0.15724865391008952], [0.6128319350712238, 0.2014526119098326, 0.2138651830581092, 0.20882533494293348, 0.0310912078688143, 0.09921089300249963, 0.0939675027431702, 0.0020116294559333653, 0.0, 0.08073682108724395], [0.48771461766368096, 0.13126538984837854, 0.2363462360681022, 0.1763634371924744, 0.0, 0.04412631033831555, 0.006806505615668833, 0.0, 0.0, 0.050108857930002255], [0.3423508198510157, 0.10310870336034031, 0.08390180751851728, 0.12647423810407488, 0.001838218572703418, 0.01027562900228901, 0.0, 0.0, 0.0, 0.0], [0.19500434034570224, 0.042510840532904784, 0.15137339875559855, 0.02276819961976874, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.12677653618212845, 0.031965656862295896, 0.03804500826428606, 0.01195444858741944, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.06828261930028688, 0.03262116316636906, 0.03317321633836416, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
ap_se_ws = [[0.9403910844273068, 0.597508316431782, 0.6628757426439243, 0.41081879785079, 0.0558210811932023, 0.44123754568493245, 0.62935956791067, 0.06392388309122063, 0.0, 0.5040457806973629], [0.8422225382356734, 0.37742818034844994, 0.4633736927542431, 0.21071435195718463, 0.0833371942633445, 0.3672796932597658, 0.2987135070252783, 0.060894015187202055, 0.0, 0.34025937389679706], [0.7639087537516625, 0.24120011670755026, 0.5404592841553006, 0.3619865109108643, 0.22761614454936052, 0.26336160689550103, 0.10206828732873456, 0.010506443123832709, 0.0, 0.365999036413526], [0.6561777288620275, 0.21585941883113763, 0.23635771008009104, 0.13070457764700075, 0.015444771445454765, 0.19112207170485607, 0.021082214551779225, 0.003483217592592593, 0.0, 0.17429226675395315], [0.5250649151701028, 0.12154165195731548, 0.22285461335844725, 0.14641576414822943, 0.0, 0.12469057861947799, 0.0, 0.0, 0.0, 0.1115338090601067], [0.3890809387617653, 0.10414509703649052, 0.09524621668342749, 0.1485135866290485, 0.0, 0.07856792731591004, 0.0, 0.0, 0.0, 0.0], [0.23411860225538467, 0.08995369167703733, 0.1663685606070418, 0.026467928318781014, 0.0, 0.02344565304289691, 0.0, 0.0, 0.0, 0.0], [0.16038544342289818, 0.025578502985855643, 0.018421731471122908, 0.004768463172474235, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.09546565690701568, 0.03539678449435006, 0.005976813456806771, 0.0011909659551255826, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]

te_baseline=[[0.1395990537520651, 0.3657720389986252, 0.2833404060833379, 0.41315852344853515, 0.7957644566695609, 0.11576769030858387, 0.15924761612063115, 1.0, 1.0, 0.31603111760375985], [0.15680059261606327, 0.2813834288046291, 0.4381869820266808, 0.5792921837223198, 0.7641330877589158, 0.12646309399554495, 0.17383410943862115, 0.1998589685767544, 1.0, 0.42245243731935], [0.19518052688613838, 0.35836205125104564, 0.3609138234957137, 0.4470205788331464, 0.5292347274526777, 0.14122651376383225, 0.22678449305966358, 0.17408224619705537, 1.0, 0.43577921298699135], [0.22284884306567948, 0.43718488030025204, 0.4963002761717784, 1.0425483146541792, 0.7930949272831167, 0.1485954057644341, 0.3718504640682626, 1.0, 1.0, 0.506021710461826], [0.24788694816536314, 0.5873136835729345, 0.6772719371094741, 1.1640118998491966, 1.0, 0.16048184783503777, 1.0, 1.0, 1.0, 0.5148830682300851], [0.2810691528993628, 0.5400353453750216, 0.5600895865230489, 0.998564648363276, 1.0, 0.1625431297964624, 1.0, 1.0, 1.0, 1.0], [0.32163412822646087, 0.5023357633194755, 0.8303151556791039, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [0.35720471133819526, 0.6045135744693921, 0.7174330296277971, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [0.3456795294512356, 0.5397549732629308, 0.7009528671910797, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]
se_baseline=[[0.3577923426321594, 0.33471034601576294, 0.33820018598003887, 0.4733017292961613, 0.5712220697338044, 0.3809077737399612, 0.3061068438803559, 1.0, 1.0, 0.5056639645009935], [0.48264529384278754, 0.3510920481227825, 0.35406119905826416, 0.439084033117139, 0.5563321473783069, 0.4632414737267041, 0.5499758579247214, 0.46734269580596893, 1.0, 0.5862868483184016], [0.5810801814666197, 0.45476541657036046, 0.4058662158364447, 0.5052518118173055, 0.5380441467283658, 0.575260778296617, 0.6756088023792371, 0.6462089759230991, 1.0, 0.6720951671513626], [0.6588173777426852, 0.5289586176125136, 0.45407266090514464, 0.5893077788408223, 0.6526304342841323, 0.6618763781455465, 0.7679745551215078, 1.0, 1.0, 0.720135723488753], [0.7291009040942994, 0.6526620050909268, 0.5226103765591171, 0.6033502855062096, 1.0, 0.7184636542349869, 1.0, 1.0, 1.0, 0.7210680204860702], [0.7674698088430061, 0.6827515200049151, 0.548221009099212, 0.6899274752955731, 1.0, 0.7462409240759145, 1.0, 1.0, 1.0, 1.0], [0.7761552496705039, 0.7299967650767141, 0.661843467032234, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [0.7819567070469845, 0.7174176946457076, 0.735241899082331, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [0.713048756508702, 0.7245079756635214, 0.6986760646890833, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]

te_target = [[0.15300955392882967, 0.25161593871498406, 0.2918325907345018, 0.6448543238541825, 0.8526331268476359, 0.12464815487117788, 0.1634706283592422, 0.13464136482092687, 1.0, 0.27647236484552984], [0.16541052762599828, 0.3072474482740126, 0.2556758164917155, 0.7243310767360455, 0.7349927154109896, 0.12320987457392599, 0.18430324660331907, 0.15877656142871083, 1.0, 0.3018671365686768], [0.17789588384013152, 0.3212234716954239, 0.4283542090532196, 0.8268579654744125, 0.5707387098264823, 0.13305812769857508, 0.20987433849938145, 0.19238550035614377, 1.0, 0.3030523500631169], [0.19547176185800577, 0.2730219766541712, 0.4441602503471879, 0.9331746106280342, 0.5029537592520112, 0.14408850844937546, 0.2814834064781424, 0.14951047407604348, 1.0, 0.3566721143922523], [0.22008122346862977, 0.4091110601131215, 0.4490254614883585, 0.9020286903084446, 0.6495313573960881, 0.15933583779033292, 0.35118544960728515, 1.0, 1.0, 0.35001472182419846], [0.23437481813218383, 0.37818316992986584, 0.44675314090380286, 0.8956905282270993, 0.510444304381533, 0.16234028788475713, 1.0, 1.0, 1.0, 1.0], [0.2712148232597215, 0.4111104026384899, 0.6529873240400255, 0.7075955281739894, 0.8092090679672757, 0.17303904584619242, 1.0, 1.0, 1.0, 1.0], [0.288346971840981, 0.49596984782586806, 0.5946360956152492, 0.9989423371730471, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [0.3012102268227773, 0.4595561857985039, 0.5253753793923179, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]
se_target = [[0.36079820067788976, 0.3122974073954939, 0.36091950861539485, 0.4312203760003754, 0.5268214713057612, 0.370384845707801, 0.31128420341709934, 0.6077073469745162, 1.0, 0.5081301361168067], [0.48742835585083244, 0.3708823598611649, 0.3539989481584219, 0.37577205961156707, 0.5503724738797847, 0.4686939496926124, 0.5688131768089789, 0.543406634989273, 1.0, 0.5956601020750866], [0.5786791585942673, 0.4558278455838345, 0.433367951067452, 0.5220987467066617, 0.5733486603071649, 0.5755340382935513, 0.6563285059956456, 0.6179702410587319, 1.0, 0.6840299487461088], [0.6454908551367109, 0.5425888781003896, 0.4607379503119907, 0.6015359180640295, 0.5781918600103332, 0.6504734435509443, 0.7925750180090295, 0.6584933807424692, 1.0, 0.731901423031489], [0.7116307796703492, 0.6368547195052894, 0.5505948204505542, 0.6452797163924864, 0.6019605044754972, 0.6929726157879317, 0.8600928994520748, 1.0, 1.0, 0.7186487581568036], [0.7476226933296167, 0.6683724363977277, 0.5466433247886523, 0.7049203183814713, 0.5881184470955468, 0.7398803746978249, 1.0, 1.0, 1.0, 1.0], [0.7542127729740176, 0.7118624490656568, 0.6805636313182116, 0.7048807019928732, 0.7143814144522975, 0.7517643563857653, 1.0, 1.0, 1.0, 1.0], [0.7493034047664889, 0.7085843254570362, 0.7577134913454499, 0.7027172907533933, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [0.6927441103596094, 0.7077308280373098, 0.746798286165364, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]

te_seq12 = [[0.13651962426036643, 0.33409362403797255, 0.27341687766407585, 0.3683908167949623, 0.8786713608808496, 0.12042787580737938, 0.1512325000792576, 0.1729131900437835, 1.0, 0.3315818194687734], [0.16708417115924495, 0.3245748916191466, 0.30785036164817847, 0.5326806204987424, 0.9958173387660793, 0.1283723867500725, 0.1482330254105102, 0.1699350115541705, 1.0, 0.37311383716164276], [0.19499086609608268, 0.3753681241735245, 0.4667943221318268, 0.6801366354313026, 0.4479293426152896, 0.1583271309649306, 0.22094827191329844, 0.23676362001341972, 1.0, 0.38694453711759164], [0.21381562567826534, 0.3754821460201112, 0.4983964740980698, 0.6514878515014821, 0.8403937467265407, 0.15017375000202002, 0.2597564886211459, 0.23815267325407077, 1.0, 0.5031232866143213], [0.2476988385433142, 0.4556775294447918, 0.6800958306486096, 1.092395303557685, 1.0, 0.1612887774443277, 1.0, 1.0, 1.0, 0.5040365067780009], [0.28229509374497425, 0.44396534831849205, 0.5747072557435098, 0.9738571074132949, 1.0, 0.16607621776263054, 1.0, 1.0, 1.0, 1.0], [0.32783972640814124, 0.5049135875724262, 0.6319221354872683, 0.8363060444901814, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [0.3841766048121086, 0.6066471257252507, 0.6455060912584892, 0.969134975687428, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [0.4064571091838607, 0.6354759256867503, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]
se_seq12 = [[0.3560519242112571, 0.31872208525158696, 0.33465631238353455, 0.4444761165516771, 0.5925349806940268, 0.3774498081844505, 0.3177149288853232, 0.5130992641791429, 1.0, 0.5150470111233327], [0.476535707649013, 0.3536682200026068, 0.35635336035495424, 0.40175119135931164, 0.6013665289750831, 0.46823398720708037, 0.5216645551487782, 0.45284541996108735, 1.0, 0.5925765875404205], [0.5777304203481691, 0.4649678349092596, 0.41705059686553003, 0.5094847471094263, 0.5563164231426362, 0.5780869506701745, 0.6838043268486972, 0.571555663585694, 1.0, 0.6495101208810282], [0.6425444696477727, 0.5193718137188305, 0.48341560954686463, 0.587726114228544, 0.6516225074994823, 0.6371069498603478, 0.7424163726780381, 0.6158383967525296, 1.0, 0.7042673061087141], [0.7169219975599971, 0.646814458887103, 0.5359427885085889, 0.6318564026062001, 1.0, 0.6793574945415171, 1.0, 1.0, 1.0, 0.6816209204064122], [0.7569161914456219, 0.6253802789660307, 0.5668834970746733, 0.672217917667572, 1.0, 0.7212181786438492, 1.0, 1.0, 1.0, 1.0], [0.7605211380214819, 0.7057659648043265, 0.645879869426182, 0.6926833616764799, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [0.7523880389266104, 0.7132626312896523, 0.7260616514666487, 0.7481535423022172, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [0.6726701257327156, 0.7235975704353992, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]

te_seq32 = [[0.1390809381663216, 0.3359296890381864, 0.22316804773709933, 0.3853362549528289, 1.082183957113216, 0.11364450726531579, 0.14543605547716085, 0.2956936202636783, 1.0, 0.3148798402886928], [0.16958668709167474, 0.3167527645225041, 0.314953758551765, 0.30692712117617776, 0.9199513639390344, 0.12211097251331073, 0.17075503656175078, 0.18336436202201947, 1.0, 0.36221123579493714], [0.19316128291049728, 0.39065230420884844, 0.49438580039798236, 0.766969414092211, 0.47096543784327805, 0.14309424952070823, 0.2433808299056172, 0.19556946492112257, 1.0, 0.39271850231269906], [0.22558196983132958, 0.4711983261576493, 0.5476186115992868, 0.7175052885336074, 0.7707662296037128, 0.15732364263661455, 0.3624169565927334, 0.28979325731756783, 1.0, 0.4729559762251475], [0.26175560469072046, 0.5229677427756065, 0.5951653486720914, 0.8256618820906317, 1.0, 0.1596518694133149, 0.365043454189, 1.0, 1.0, 0.5362458293265586], [0.28793190902470595, 0.4921931620989657, 0.5283070947575373, 0.946720046612751, 1.0, 0.16561065107507897, 1.0, 1.0, 1.0, 1.0], [0.32342895528706855, 0.5989139073037828, 0.4966133384171786, 0.7764040848927568, 1.0, 0.17927930454638816, 1.0, 1.0, 1.0, 1.0], [0.3696308549246056, 0.623895116928952, 0.7650365053415543, 0.9327727757964586, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [0.3686915090660646, 0.5591717484307783, 0.5058897000033447, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]
se_seq32= [[0.3598862784226796, 0.3422038341606482, 0.3289658317103841, 0.4284687106207666, 0.5320525750782099, 0.38741796928079575, 0.3091023329433759, 0.6212780211716437, 1.0, 0.5065866937739869], [0.4903042335294562, 0.3525860352911563, 0.3507051251955965, 0.35979940388139225, 0.5811787818020381, 0.47613115130320754, 0.562444343730033, 0.4816601842716236, 1.0, 0.5860507305096994], [0.5837925315002401, 0.4633767986342559, 0.43602239102237084, 0.5338245231685985, 0.5418847853276816, 0.5782294971498083, 0.6735516294567219, 0.583664414636261, 1.0, 0.6779516895755725], [0.6515130561353253, 0.540067439599715, 0.5004685613449751, 0.6043579422017153, 0.6637448007790471, 0.6420951780491538, 0.7552132875697347, 0.6835375693612663, 1.0, 0.7120583966378257], [0.7090057903932141, 0.6441642867357658, 0.5321621799897114, 0.6368819127144513, 1.0, 0.6738594342350704, 0.8188982320727561, 1.0, 1.0, 0.7006238620751478], [0.7517296585933428, 0.6685300940492066, 0.555924346244278, 0.6925544088376324, 1.0, 0.6935792627746195, 1.0, 1.0, 1.0, 1.0], [0.75511999955775, 0.7219395700045793, 0.6692013155485755, 0.6902885133613086, 1.0, 0.7451126582147007, 1.0, 1.0, 1.0, 1.0], [0.7651050825405272, 0.7317258857230741, 0.7343488433374727, 0.7415663051067289, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [0.6935666810976704, 0.7369001287396119, 0.7183401314913208, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]

te_se = [[0.14185766065652022, 0.24926020889760783, 0.23135489714828902, 0.3136819458608806, 1.0, 0.11926429492755299, 0.1380508654692559, 0.14897188954154642, 1.0, 0.27997886276185907], [0.17022510916548156, 0.3893250355759534, 0.32088192031845714, 0.40341989186454014, 0.8379176433503654, 0.1242741428461431, 0.16088744855245024, 0.18187379275893537, 1.0, 0.3146098351838853], [0.20195977991344502, 0.40617021888337973, 0.3749827968515561, 0.6561752689508271, 0.6494025081138636, 0.13769181306449846, 0.21638215901594238, 0.18007358330165404, 1.0, 0.37471910689146065], [0.2278200700810846, 0.4439272009761229, 0.4309380899903147, 0.6407705805486139, 0.7731755005726554, 0.1457676660949315, 0.3589252972801193, 0.21190691182829802, 1.0, 0.45319779082299655], [0.25257639814289484, 0.614373626521685, 0.5770929214275483, 0.9407423971084282, 1.0, 0.15829236168950794, 0.38712828095825136, 1.0, 1.0, 0.5398412313195502], [0.2817134210411889, 0.42355819121681243, 0.48725228277318566, 0.7296461519102292, 1.0, 0.17484230489002706, 0.4049262102634358, 1.0, 1.0, 1.0], [0.33067574950064654, 0.5388241458482548, 0.6686314861866737, 0.8658641391371709, 1.0, 0.17952334078027737, 1.0, 1.0, 1.0, 1.0], [0.3834453520082466, 0.6438572338303389, 0.6829229369351729, 0.8200651263869142, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [0.40815760711293636, 0.6133857123572137, 1.0, 0.9122870576778269, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]
se_se = [[0.3506025601558952, 0.32399114625394787, 0.3343822072625803, 0.49744828868497204, 1.0, 0.38181691060598694, 0.29664671354461614, 0.5855459480370734, 1.0, 0.49590642948851577], [0.4768989560122449, 0.36180226759080786, 0.34875881889879234, 0.35061784486838027, 0.5887632778359162, 0.47047821160671244, 0.5414919674273388, 0.4944412678251303, 1.0, 0.5771787821805335], [0.576694360869871, 0.4587876578421657, 0.4270868526055213, 0.48574278490470374, 0.5937518737939853, 0.5707936387243817, 0.6486858486696261, 0.5815238005165413, 1.0, 0.6911354325945793], [0.6587395032887245, 0.5221801776525518, 0.4609928337355895, 0.6034379103248058, 0.6649134060101596, 0.6470580661081905, 0.7653450124040344, 0.6007269724970383, 1.0, 0.7111690248612178], [0.7090651391942279, 0.635296963049327, 0.5127217581947658, 0.6392177138601546, 1.0, 0.6978059851242718, 0.8027404904652754, 1.0, 1.0, 0.6814349352975241], [0.7382312588714818, 0.6782163388524219, 0.5405943299003202, 0.687881803873021, 1.0, 0.7474737031993658, 0.7978772582718574, 1.0, 1.0, 1.0], [0.7412257674898324, 0.7144802608320026, 0.6365286584779156, 0.684735191579922, 1.0, 0.7492427988429127, 1.0, 1.0, 1.0, 1.0], [0.7395044450032594, 0.7272242087214192, 0.7114447744791267, 0.7559256626696191, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [0.6548165235537571, 0.7338068156637229, 1.0, 0.7998305432873284, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]

te_se_extra = [[0.14239138775955945, 0.24895301464825464, 0.2676623683716511, 0.3721533428440034, 0.7607812883725191, 0.11444803783094712, 0.14423447233589662, 1.0, 1.0, 0.26832394503221835], [0.1594870043151833, 0.3408970896499083, 0.4006771018316545, 0.8792139037515235, 0.7593708350205844, 0.11952455060986425, 0.16563200072181863, 0.1837097349165101, 1.0, 0.3382381571813762], [0.19712199158034446, 0.36017117271819726, 0.4076317730733835, 0.699069202766759, 0.5418767900286311, 0.1337212343804098, 0.21925490327657693, 0.2500342847301834, 1.0, 0.3832738087259042], [0.22949169512772238, 0.4143853888801443, 0.6177891291286833, 1.1212529736947199, 0.8547664580535079, 0.14452360337796247, 0.3366653726501072, 0.308057068638183, 1.0, 0.486399144208533], [0.2611586973864331, 0.4995158911089175, 0.5963671373344879, 0.8717918206209023, 1.0, 0.1485689527533807, 0.42583364984684985, 1.0, 1.0, 0.49410934127911765], [0.29554272747903804, 0.48095631545550827, 0.6163902675062745, 1.0196947666971963, 0.8182056201591758, 0.16363265785333855, 1.0, 1.0, 1.0, 1.0], [0.33551339771634064, 0.6215062720164524, 0.4212626725074527, 0.9157416875002912, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [0.3873924062282604, 0.6005387290565455, 0.7150607019973206, 1.03357849191646, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [0.42545243343676137, 0.5415325264058723, 0.6591348631155701, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]
se_se_extra = [[0.3612369444296266, 0.312321834361407, 0.3232653658511183, 0.4873627898242925, 0.5774368037791721, 0.38504736348174123, 0.33060834889170615, 1.0, 1.0, 0.4948808469577295], [0.48963434253833454, 0.34916212078461006, 0.34402696611856376, 0.3764670783013008, 0.5822494269222982, 0.4569549667656707, 0.5209045471852806, 0.4643342716301356, 1.0, 0.5804650186080821], [0.5792934255680091, 0.44454746296357434, 0.40153182155371514, 0.4978106554927163, 0.5799153001323368, 0.5558564902502772, 0.6547988580338712, 0.5648605974132823, 1.0, 0.6940094020302173], [0.659977076536846, 0.5125117904820116, 0.48019010439492654, 0.5912346862985164, 0.6371256738687794, 0.6342505728499757, 0.7755563333081718, 0.607380628214772, 1.0, 0.736146237946219], [0.7287726745000844, 0.6581395904873023, 0.5332156936339671, 0.6257725418847025, 1.0, 0.6765246694211263, 0.8443464763797728, 1.0, 1.0, 0.7275339807646899], [0.7619159278071131, 0.666767846078413, 0.5502165342844685, 0.7111506824015777, 0.5928704624439138, 0.7320851130754032, 1.0, 1.0, 1.0, 1.0], [0.7686333974640372, 0.7107632738015406, 0.6540683339886567, 0.7040562005893914, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [0.7842506692706089, 0.7252940167424654, 0.7070484633093481, 0.7088309273785183, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [0.6823950431563676, 0.7190695825128153, 0.7812863557382507, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]

te_se_ws = [[0.1417589460571254, 0.36188566431726105, 0.361019880627202, 0.2808253559213234, 0.6886119402156241, 0.11942617683736412, 0.13742864631272173, 0.1306409309894744, 1.0, 0.3188863179130802], [0.1651648483952256, 0.32437989855265575, 0.3201913613747657, 0.39022658701086843, 0.8383555276960942, 0.1264969544057468, 0.1860380077234916, 0.17239745098468126, 1.0, 0.396010471677784], [0.1965428589112319, 0.38145320916825304, 0.35134863310695175, 0.5564372122624913, 0.4885700177201704, 0.14277414702295058, 0.2338669752685044, 1.0, 1.0, 0.40650352239096377], [0.22254016871243912, 0.43044679026720145, 0.46385310726566964, 1.0997886246213564, 0.8820788474250418, 0.14658418305119603, 0.3220226468624505, 1.0, 1.0, 0.5294884613226187], [0.24526216840483062, 0.6744050688905674, 0.6200542660105496, 1.3208062312507476, 1.0, 0.15837150193377145, 1.0, 1.0, 1.0, 0.5511659144299728], [0.2786480011268999, 0.5016148415037495, 0.6621190775435601, 0.9165254986892697, 1.0, 0.17223029427820463, 1.0, 1.0, 1.0, 1.0], [0.3264577717280904, 0.5413453476246174, 0.7144091574679824, 0.8787820270968685, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [0.36327599204858413, 0.5970049618624698, 0.7542009324299571, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [0.3764646291411952, 0.5294328376510244, 0.6645721930343401, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]
se_se_ws = [[0.3492686772409399, 0.35713483187651285, 0.3270041853511598, 0.434334657891002, 0.569337844760188, 0.37604673875322714, 0.30076322162581437, 0.5777454281771894, 1.0, 0.5006404552926174], [0.4852737990631568, 0.342840875577621, 0.35289323546803114, 0.3963386277473872, 0.5606807814078012, 0.4602069079249741, 0.5195352562894593, 0.48258814758370516, 1.0, 0.5842918615706613], [0.5837517785490958, 0.455755008688267, 0.40806849844188786, 0.47016188142089127, 0.5785522568910427, 0.5759546749906987, 0.6525170821588082, 1.0, 1.0, 0.6925794684155554], [0.6498687274896411, 0.5059887771766214, 0.4542257711033549, 0.5833015279386431, 0.6168142275156436, 0.6531673340944576, 0.7753980634382461, 1.0, 1.0, 0.7433163550275472], [0.7264686650585723, 0.6874006816362667, 0.5362798919236639, 0.6323365495561025, 1.0, 0.7009916647217095, 1.0, 1.0, 1.0, 0.7397056763383548], [0.7626054474475411, 0.679172586590263, 0.5597472088115262, 0.6868617716820722, 1.0, 0.7472416684218278, 1.0, 1.0, 1.0, 1.0], [0.7795491505606525, 0.7531624291260326, 0.6736722888475836, 0.6999941018682169, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [0.7775392506971647, 0.7245791912073536, 0.6979880402279105, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [0.6898978142928281, 0.7161683279466854, 0.722235284326462, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]

ap_baseline_risc = np.asarray(list(zip(*ap_baseline)))
ap_se_risc = np.asarray(list(zip(*ap_se)))
ap_seq12_risc = np.asarray(list(zip(*ap_seq12)))
ap_seq32_risc = np.asarray(list(zip(*ap_seq32)))
ap_target_risc = np.asarray(list(zip(*ap_target)))
ap_se_extra_risc = np.asarray(list(zip(*ap_se_extra)))
ap_se_ws_risc = np.asarray(list(zip(*ap_se_ws)))

te_baseline_risc = np.asarray(list(zip(*te_baseline)))
te_se_risc = np.asarray(list(zip(*te_se)))
te_seq12_risc = np.asarray(list(zip(*te_seq12)))
te_seq32_risc = np.asarray(list(zip(*te_seq32)))
te_target_risc = np.asarray(list(zip(*te_target)))
te_se_extra_risc = np.asarray(list(zip(*te_se_extra)))
te_se_ws_risc = np.asarray(list(zip(*te_se_ws)))

se_baseline_risc = np.asarray(list(zip(*se_baseline)))
se_se_risc = np.asarray(list(zip(*se_se)))
se_seq12_risc = np.asarray(list(zip(*se_seq12)))
se_seq32_risc = np.asarray(list(zip(*se_seq32)))
se_target_risc = np.asarray(list(zip(*se_target)))
se_se_extra_risc = np.asarray(list(zip(*se_se_extra)))
se_se_ws_risc = np.asarray(list(zip(*se_se_ws)))




meta_architectures_num = 7

comparison_all = np.zeros(shape=[10,meta_architectures_num,9])
comparison_all_te = np.zeros(shape=[10,meta_architectures_num,9])
comparison_all_se = np.zeros(shape=[10,meta_architectures_num,9])

for objClassIndex in range(10):
    # comparison_per_class = np.concatenate((np.asarray(ap_baseline_risc[objClassIndex]),ap_seq32_risc[objClassIndex],ap_se_risc[objClassIndex]), axis=0)
    comparison_all[objClassIndex] = np.vstack((np.asarray(ap_baseline_risc[objClassIndex]),ap_target_risc[objClassIndex],ap_seq12_risc[objClassIndex], ap_seq32_risc[objClassIndex],ap_se_risc[objClassIndex], ap_se_extra_risc[objClassIndex], ap_se_ws_risc[objClassIndex]))
    comparison_all_te[objClassIndex] = np.vstack((np.asarray(te_baseline_risc[objClassIndex]), te_target_risc[objClassIndex],
                                      te_seq12_risc[objClassIndex], te_seq32_risc[objClassIndex],
                                      te_se_risc[objClassIndex], te_se_extra_risc[objClassIndex],
                                      te_se_ws_risc[objClassIndex]))
    comparison_all_se[objClassIndex] = np.vstack((np.asarray(se_baseline_risc[objClassIndex]), se_target_risc[objClassIndex],
                                      se_seq12_risc[objClassIndex], se_seq32_risc[objClassIndex],
                                      se_se_risc[objClassIndex], se_se_extra_risc[objClassIndex],
                                      se_se_ws_risc[objClassIndex]))

    # print(ap_baseline_risc[objClassIndex].shape)
    # comparison_all[objClassIndex] = comparison_per_class


# for row1,row2,row3 in ap_baseline_risc,ap_seq32_risc,ap_se_risc
#     classCompararison = np.concatenate((row1,row2,row3))

x_grids = [5, 12.5, 17.5, 22.5, 27.5, 32.5, 37.5, 42.5, 47.5]

# style
plt.style.use('seaborn-darkgrid')
# create a color palette
palette = plt.get_cmap('Set1')
# plt.plot(x_grids, ap_car, marker='', linewidth=1, alpha=0.9, label='ap_car')

string = ['car', 'truck', 'bus', 'trailer', 'construction_vehicle', 'pedestrian', 'motorcycle', 'bicycle',
          'traffic_cone', 'barrier']

# multiple line plot
for objClassIndex in range(10):

    string_meta = ['Baseline', 'Target', 'Seq-12', 'Seq-32', 'SE', 'SE_extra', 'SE_shared']
    num = 0
    for i in range(meta_architectures_num):
        plt.plot(x_grids, comparison_all[objClassIndex][i], marker='', color=palette(num), linewidth=1, alpha=0.9, label=string_meta.pop(0))
        num += 1
    # Add legend
    plt.legend(loc='upper right')
    # Add titles
    classNam=string.pop(0)
    plt.title("AP for {} with respect to detection range".format(classNam), loc='center', fontsize=12, fontweight=0, color='black')
    plt.xlabel("Detection Range (m)")
    plt.ylabel("AP")


    plt.draw()
    plt.savefig(os.path.join("/mrtstorage/projects/grid_map_learning/nuScenes_object_detection/model_dirs", '{}_AP_comparison.pdf'.format(classNam)),bbox_inches='tight')
    plt.close()




comparison_LARGE = (comparison_all[0] + comparison_all[1] + comparison_all[2] + comparison_all[3] + comparison_all[4]) / 5

string_meta = ['baseline', 'target', 'sequential-12', 'sequential-32', 'shared_encoder', 'shared_encoder_extraExtractor', 'shared_encoder_weightSharing']
num = 0
for i in range(meta_architectures_num):
    plt.plot(x_grids, comparison_LARGE[i], marker='', color=palette(num), linewidth=1, alpha=0.9, label=string_meta.pop(0))
    num += 1
# Add legend
plt.legend(loc='upper right')
# Add titles

plt.title("AP for LARGE objects with respect to detection range", loc='center', fontsize=12, fontweight=0, color='black')
plt.xlabel("Detection Range (m)")
plt.ylabel("AP")


plt.draw()
plt.savefig(os.path.join("/mrtstorage/projects/grid_map_learning/nuScenes_object_detection/model_dirs", 'LARGE_AP_comparison.pdf'),bbox_inches='tight')
plt.close()











comparison_SMALL = (comparison_all[5] + comparison_all[6] + comparison_all[7] + comparison_all[8] + comparison_all[9]) / 5

string_meta = ['baseline', 'target', 'sequential-12', 'sequential-32', 'shared_encoder', 'shared_encoder_extraExtractor', 'shared_encoder_weightSharing']
num = 0
for i in range(meta_architectures_num):
    plt.plot(x_grids, comparison_SMALL[i], marker='', color=palette(num), linewidth=1, alpha=0.9, label=string_meta.pop(0))
    num += 1
# Add legend
plt.legend(loc='upper right')
# Add titles

plt.title("AP for SMALL objects with respect to detection range", loc='center', fontsize=12, fontweight=0, color='black')
plt.xlabel("Detection Range (m)")
plt.ylabel("AP")

plt.tight_layout()

plt.draw()
plt.savefig(os.path.join("/mrtstorage/projects/grid_map_learning/nuScenes_object_detection/model_dirs", 'SMALL_AP_comparison.pdf'),bbox_inches='tight')
plt.close()














comparison_ALLES = (comparison_all[0] + comparison_all[1] + comparison_all[2] + comparison_all[3] + comparison_all[4] + comparison_all[5] + comparison_all[6] + comparison_all[7] + comparison_all[8] + comparison_all[9]) / 10

string_meta = ['baseline', 'target', 'sequential-12', 'sequential-32', 'shared_encoder', 'shared_encoder_extraExtractor', 'shared_encoder_weightSharing']
num = 0
for i in range(meta_architectures_num):
    plt.plot(x_grids, comparison_ALLES[i], marker='', color=palette(num), linewidth=1, alpha=0.9, label=string_meta.pop(0))
    num += 1
# Add legend
plt.legend(loc='upper right')
# Add titles

plt.title("AP for all objects with respect to detection range", loc='center', fontsize=12, fontweight=0, color='black')
plt.xlabel("Detection Range (m)")
plt.ylabel("AP")


plt.draw()
plt.savefig(os.path.join("/mrtstorage/projects/grid_map_learning/nuScenes_object_detection/model_dirs", 'ALL_AP_comparison.pdf'),bbox_inches='tight')
plt.close()




comparison_ALLES_te = (comparison_all_te[0] + comparison_all_te[1] + comparison_all_te[2] + comparison_all_te[3] + comparison_all_te[4] + comparison_all_te[5] + comparison_all_te[6] + comparison_all_te[7] + comparison_all_te[8] + comparison_all_te[9]) / 10

string_meta = ['baseline', 'target', 'sequential-12', 'sequential-32', 'shared_encoder', 'shared_encoder_extraExtractor', 'shared_encoder_weightSharing']
num = 0
for i in range(meta_architectures_num):
    plt.plot(x_grids, comparison_ALLES_te[i], marker='', color=palette(num), linewidth=1, alpha=0.9, label=string_meta.pop(0))
    num += 1
# Add legend
plt.legend(loc='upper left')
# Add titles

plt.title("ATE for all objects with respect to detection range", loc='center', fontsize=12, fontweight=0, color='black')
plt.xlabel("Detection Range (m)")
plt.ylabel("ATE")


plt.draw()
plt.savefig(os.path.join("/mrtstorage/projects/grid_map_learning/nuScenes_object_detection/model_dirs", 'ALL_ATE_comparison.pdf'),bbox_inches='tight')
plt.close()








comparison_ALLES_se = (comparison_all_se[0] + comparison_all_se[1] + comparison_all_se[2] + comparison_all_se[3] + comparison_all_se[4] + comparison_all_se[5] + comparison_all_se[6] + comparison_all_se[7] + comparison_all_se[8] + comparison_all_se[9]) / 10

string_meta = ['baseline', 'target', 'sequential-12', 'sequential-32', 'shared_encoder', 'shared_encoder_extraExtractor', 'shared_encoder_weightSharing']
num = 0
for i in range(meta_architectures_num):
    plt.plot(x_grids, comparison_ALLES_se[i], marker='', color=palette(num), linewidth=1, alpha=0.9, label=string_meta.pop(0))
    num += 1
# Add legend
plt.legend(loc='upper left')
# Add titles

plt.title("ASE for all objects with respect to detection range", loc='center', fontsize=12, fontweight=0, color='black')
plt.xlabel("Detection Range (m)")
plt.ylabel("ASE")


plt.draw()
plt.savefig(os.path.join("/mrtstorage/projects/grid_map_learning/nuScenes_object_detection/model_dirs", 'ALL_ASE_comparison.pdf'),bbox_inches='tight')
plt.close()