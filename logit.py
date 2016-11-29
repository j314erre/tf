import operator
import numpy
from scipy import optimize as opt
from scipy import stats
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import r2_score
from collections import defaultdict

#Logistic function
def logistic(z):
    return 1.0 / (1.0 + numpy.exp(-z))


#Predict Function
def predict(w, x):
    return logistic(numpy.dot(w,x))


#Calculate Cross Entropy Loss
def cross_entropy_loss(X,Y,w):
    #p = {true_click_prob, true_no_click_prob} or {y, 1-y}
    #q = {predicted_click_prob, predicted_no_click_prob} or {y_hat, 1-y_hat}
    #Cross Entropy - H(p,q) = -sum(i)p_i * log q_i = -y log y_hat - (1-y)log(1-y_hat) 
    N = len(X)
    if type(w) == tuple:
        y_hat = logistic(numpy.dot(X,w[0]))
    else:
        y_hat = logistic(numpy.dot(X,w))
    return -1.0/N * numpy.sum(Y * numpy.log(y_hat) + (1-Y)*numpy.log(1-y_hat))


def entropy(Y):
    N = len(Y)
    return -1.0/N * numpy.sum(Y*numpy.log(Y)+(1-Y)*numpy.log(1-Y))


# Function that trains w using BFGS minimization
def train_w(X, Y, std=0.1):

    def f(w):
        return cross_entropy_loss(X,Y,w)
    
    K = X.shape[1]
    initial_guess = numpy.random.normal(0,std,K)
    return opt.fmin_l_bfgs_b(f, initial_guess, approx_grad=1,disp=0)


# Evaluation
def error(X,Y,w):
    y_predict = []
    for i in range(len(X)):
        y_predict.append(predict(X[i],w))
    #uncomment to show line by line predictions
    # for x,y in zip(Y,y_predict):
    #     print x,'\t',y
    return mean_squared_error(Y,y_predict)


def KL_divergence(X,Y,w):
    return cross_entropy_loss(X,Y,w) - entropy(Y)


#Calculates average KL divergence
def avg_error(all_X, all_Y, std):
    print 'training weight std: ', std
    s = 0
    K = len(all_X)
    for i in range(K):
        X_heldout, X_rest = all_X[i]
        Y_heldout, Y_rest = all_Y[i]
        w = train_w(X_rest, Y_rest, std)
        s += KL_divergence(X_heldout, Y_heldout, w)   
    return s * 1.0 / K


#Trains parameter std for model
def train_w_init_std(X, Y, K=10):
    all_std = numpy.array([0.01,0.03,0.1,0.3,1,3,10,30,100])  #suggested options by Richardson 2009 paper
    all_X = kfold(X, K)
    all_Y = kfold(Y, K)
    all_err = numpy.array([avg_error(all_X, all_Y, std) for std in all_std])
    return all_std[all_err.argmin()]


#loads training data. Processes training data, normalizing feature vectors. Loads Test data and normalizes Test data based on training feature means and standard deviations
def read_data(f_input,sep="\t",num_words=100):

    def split_line(line):
        return line.split(sep)

    def apply_filt(filt,values):
        return map(filt, values)

    def extract_feature_from_col(lines,col_num):
        return split_line(lines.strip())[col_num]

    def extract_y(lines):
        return split_line(lines.strip())[0]

    def extract_avg_np_cart_feature(lines):
        return split_line(lines.strip())[8]

    def extract_cat_avg_np_cart_feature(lines):
        return split_line(lines.strip())[9]

    def extract_price(lines):
        return split_line(lines.strip())[10]

    def extract_num_words(lines):
        return split_line(lines.strip())[15]

    def extract_gender(lines):
        return split_line(lines.strip())[22]

    def extract_kwd(lines):
        return split_line(lines.strip())[18]

    def extract_num_benefits(lines):
        return split_line(lines.strip())[30]

    def extract_benefit_intro(lines):
        return split_line(lines.strip())[28]

    def extract_problem_intro(lines):
        return split_line(lines.strip())[24]

    def extract_feature_count(lines):
        return split_line(lines.strip())[31]

    def extract_num_adj(lines):
        return split_line(lines.strip())[33]

    def extract_num_male_words(lines):
        return split_line(lines.strip())[36]

    def extract_spelling_score(lines):
        return split_line(lines.strip())[16]

    def extract_originality_score(lines):
        return split_line(lines.strip())[17]

    def extract_readability_score(lines):
        return split_line(lines.strip())[19]

    def extract_readability2_score(lines):
        return split_line(lines.strip())[20]

    def extract_age_score(lines):
        return split_line(lines.strip())[21]

    def extract_pageviews(lines):
        return split_line(lines.strip())[1]

    def extract_text(lines):
        return split_line(lines.strip())[11]



    def read_file_lines(filename):
        f = open(filename)
        lines = f.readlines()
        f.close()
        return lines

    def get_derived_feature_log(v):
        return [numpy.log(x+1) for x in v]

    def get_derived_feature_squared(v):
        return [x**2 for x in v]

    def select_features(lines, num_words):
        
        kpi_sum = 0.0
        kpi_average = 0.0
        kpi_count = 0.0
        word_counts = defaultdict(int)
        
        #print "READ LINES: " + str(len(lines))
        
        for line in lines:
            #columns = line.split(sep)
            kpi = extract_y(line)
            kpi_sum += float(kpi)
            kpi_count += 1.0
            text = extract_text(line)
            words = text.split(' ')
            for word in words:
                word_counts[word.lower()] += 1
                
        sorted_words = sorted(word_counts.items(), key=operator.itemgetter(1), reverse=True)
         
        best_words = dict(sorted_words[:num_words]).keys()
        #print best_words
        if kpi_count > 0:
            kpi_average = kpi_sum / kpi_count
            
        return kpi_average, best_words
            
    def extract_features(lines, baseline, words):
        x0 = [1] * len(lines) #bias feature set to 1

        #Baseline feature, average avg_np_cart
        fb = [baseline] * len(lines)
        
        
        # f0 = category average np_cart
        #f0 = apply_filt(float,map(extract_cat_avg_np_cart_feature,lines))

       

        ########### WORD FEATURES ################

        # #top_trainingset_unigrams_incl_stop_words = [u'000', u'001517', u'002552', u'003115', u'003139', u'009421', u'01', u'026m', u'029p', u'05', u'080', u'09165840', u'09779710', u'09816590', u'10', u'100', u'1000', u'10001', u'1000mm', u'1000t', u'1003mm', u'100dmtk', u'100mm', u'1010mm', u'1012mm', u'1017mm', u'101mm', u'102', u'1025mm', u'102mm', u'103', u'1030mm', u'103b', u'104', u'1040mm', u'1048', u'105', u'1050mm', u'1050rpm', u'105mm', u'1060mm', u'1063mm', u'1068mm', u'106mm', u'1070mm', u'1075mm', u'1077mm', u'108', u'1080', u'1080mm', u'1080p', u'1083mm', u'109', u'1090mm', u'10kg', u'10n6f', u'11', u'110', u'1100', u'1100mm', u'1105mm', u'112', u'1125mm', u'1128mm', u'112mm', u'113mm', u'114', u'1145mm', u'1147mm', u'1148mm', u'1150mm', u'1153mm', u'116', u'1168mm', u'1175mm', u'1178mm', u'119', u'1190mm', u'11mm', u'12', u'120', u'1200', u'1200dkb', u'1200dtkb', u'1200mm', u'1200rpm', u'120mm', u'121', u'1210w', u'122', u'1230mm', u'124', u'1243mm', u'125', u'125mm', u'127mm', u'128', u'1280', u'12800', u'128mm', u'129', u'12s2', u'12x', u'13', u'130', u'1300', u'1300mm', u'130mm', u'131mm', u'132', u'1325mm', u'1330', u'133mm', u'1351mm', u'135mm', u'136', u'1366', u'137', u'138', u'138mm', u'1390mm', u'139mm', u'14', u'140', u'1400', u'1400rpm', u'140mm', u'141', u'141mm', u'1425', u'142mm', u'1430', u'1433mm', u'144', u'1440', u'1440mm', u'145', u'1453mm', u'145mm', u'146', u'1491mm', u'15', u'150', u'1500', u'150mm', u'151', u'152mm', u'153', u'1545mm', u'156mm', u'1580mm', u'1595mm', u'16', u'160', u'1600', u'1600mm', u'1600rpm', u'160mm', u'161mm', u'162mm', u'163', u'1630mm', u'1640mm', u'1645mm', u'165mm', u'166', u'168', u'1690mm', u'1695mm', u'16mm', u'17', u'170', u'1700mm', u'170mm', u'1714mm', u'1715mm', u'1720mm', u'1730mm', u'1737mm', u'174mm', u'1756mm', u'175mm', u'176', u'1761mm', u'1774mm', u'1777mm', u'1780mm', u'1782mm', u'1785mm', u'1789mm', u'1790mm', u'17mm', u'18', u'180', u'1800', u'1800mm', u'1810b', u'181mm', u'1830mm', u'183mm', u'184mm', u'1850mm', u'185mm', u'187mm', u'188', u'188mm', u'18941whiau', u'18x', u'19', u'190mm', u'191', u'1920', u'1928', u'192mm', u'195mm', u'196', u'197mm', u'198', u'1980', u'19840au', u'1bak', u'1mm', u'1pbf1', u'1pbf64', u'20', u'200', u'2000', u'200mm', u'201779', u'202106', u'203b', u'203mm', u'204', u'204mm', u'205', u'2053', u'206273', u'2070', u'207mm', u'20x', u'210', u'2100', u'210mm', u'214', u'214mm', u'215', u'2160', u'2177f', u'218', u'21x', u'22', u'220', u'2200', u'220cl', u'222', u'222mm', u'225', u'225mm', u'227mm', u'22909', u'229mm', u'22mt45d', u'22mt55d', u'23', u'230', u'2300', u'230mm', u'2325', u'2330', u'2350', u'236', u'238mm', u'23mm', u'24', u'240', u'2400', u'240mm', u'240v', u'2420740', u'243', u'2448', u'245', u'2450', u'245mm', u'2466408', u'2466408s', u'2466409', u'2474407', u'2474409', u'2474409s', u'24751', u'2476290', u'247mm', u'24d33', u'24mm', u'24x', u'25', u'250', u'2501', u'250mm', u'251bk', u'251c', u'251cl', u'251y', u'2530011', u'2530012', u'2530013', u'2539082', u'2539084', u'2539088', u'253mm', u'255', u'2557257', u'2557259', u'255c', u'255mm', u'255y', u'2560', u'2561570', u'2561960', u'258mm', u'26', u'260', u'2600', u'260mm', u'2635654', u'264mm', u'265mm', u'266', u'267', u'268mm', u'269', u'26mm', u'27', u'270', u'2700dw', u'270mm', u'271mm', u'275mm', u'276mm', u'27mm', u'27mt55d', u'27x', u'28', u'280', u'2800', u'280mm', u'281mm', u'283mm', u'284mm', u'285mm', u'287mm', u'288mm', u'289mm', u'28mm', u'29', u'290mm', u'291mm', u'294', u'294mm', u'295mm', u'297mm', u'298', u'298mm', u'299mm', u'29c', u'29co', u'29gy', u'29pc', u'29r', u'29y', u'2g', u'2mm', u'30', u'300', u'3000', u'300111', u'300496', u'300mm', u'301201', u'301216', u'301218', u'301232', u'301985', u'302mm', u'305mm', u'306', u'306mm', u'306nw', u'307', u'307mm', u'308', u'308mm', u'3090', u'30fps', u'30mm', u'30x', u'310mm', u'313', u'313mm', u'314mm', u'315mm', u'316', u'3170cdw', u'317mm', u'32', u'320', u'320mm', u'321mm', u'323mm', u'325mm', u'3264', u'326mm', u'32a400a', u'32d33', u'32lb5610', u'32lb563b', u'330mm', u'336', u'34', u'340', u'340mm', u'343', u'345mm', u'347mm', u'34mm', u'35', u'350', u'350mm', u'352mm', u'353', u'355mm', u'35mm', u'360', u'360mm', u'368mm', u'37', u'370', u'370mm', u'374', u'375', u'375mm', u'376', u'376mm', u'377mm', u'38', u'380', u'380mm', u'3840', u'385', u'388mm', u'389', u'39', u'390', u'390mm', u'395', u'395mm', u'396', u'396mm', u'397', u'397mm', u'39mm', u'39p400m', u'39s500m', u'3d', u'3g', u'3mm', u'3x', u'40', u'400', u'400mm', u'403', u'405mm', u'409', u'40k20p', u'40k390pa', u'40mm', u'40x', u'410mm', u'411mm', u'415', u'416mm', u'41mm', u'42', u'420', u'420mm', u'423mm', u'42a400a', u'42as640a', u'42lb5610', u'42mm', u'42x', u'430', u'430mm', u'435mm', u'436', u'439mm', u'43mm', u'440', u'440mm', u'441', u'442', u'442mm', u'443', u'443mm', u'444', u'445mm', u'447', u'448mm', u'450', u'450mm', u'450upl', u'450uwl', u'454mm', u'45mm', u'460mm', u'462mm', u'469', u'46mm', u'47', u'470mm', u'475mm', u'478mm', u'48', u'480', u'4800', u'480mm', u'481mm', u'485mm', u'486mm', u'48mm', u'490mm', u'4910aub', u'494', u'495mm', u'496mm', u'49mm', u'4g', u'4mm', u'4x', u'50', u'500', u'500mm', u'501mm', u'505', u'505mm', u'50a430a', u'50as640a', u'50lb5610', u'50mm', u'510', u'510mm', u'511', u'511mm', u'514', u'515mm', u'517', u'519', u'519mm', u'520', u'520mm', u'522mm', u'523mm', u'525', u'525mm', u'526', u'526mm', u'527mm', u'52mm', u'530', u'53000', u'530mm', u'533mm', u'534mm', u'535mm', u'537mm', u'539mm', u'540', u'540mm', u'543mm', u'544mm', u'545mm', u'547mm', u'548mm', u'54mm', u'55', u'550', u'550mm', u'551mm', u'554', u'554mm', u'555mm', u'557mm', u'55k20pg', u'55k390pad', u'55mm', u'55ub820t', u'560mm', u'565mm', u'566mm', u'567mm', u'56mm', u'57', u'570mm', u'571mm', u'573mm', u'574mm', u'575', u'575mm', u'576mm', u'578mm', u'57mm', u'58', u'580', u'580mm', u'582mm', u'585mm', u'588mm', u'58mm', u'59', u'590', u'590mm', u'592mm', u'594mm', u'595mm', u'596mm', u'597mm', u'598', u'598mm', u'599mm', u'59mm', u'5diiib', u'5diiipk', u'5diiiprok', u'5kg', u'5mm', u'5x', u'60', u'600', u'6000', u'600mm', u'601', u'602mm', u'604mm', u'606mm', u'608mm', u'60as640a', u'60lb6500', u'60mm', u'61', u'610', u'610mm', u'611mm', u'612mm', u'614', u'615', u'615mm', u'617mm', u'619mm', u'61mm', u'62', u'620mm', u'624', u'625mm', u'626mm', u'629mm', u'63', u'630', u'630mm', u'634mm', u'635mm', u'636', u'637mm', u'638mm', u'64', u'640', u'640mm', u'641', u'643mm', u'645mm', u'647mm', u'64957', u'64mm', u'65', u'650', u'65022', u'65027', u'65029', u'650mm', u'657mm', u'65ax800a', u'65mm', u'65ub950t', u'65ub980t', u'66', u'660mm', u'661mm', u'662mm', u'665mm', u'668mm', u'66mm', u'67', u'670', u'670036', u'670073', u'670mm', u'671mm', u'672mm', u'675mm', u'676', u'67mm', u'68', u'680', u'680455', u'680631', u'680mm', u'685mm', u'686mm', u'68mm', u'69', u'690', u'690mm', u'695mm', u'698mm', u'6mm', u'6s1', u'6x', u'70', u'700', u'700mm', u'700rpm', u'702mm', u'703mm', u'705mm', u'709mm', u'70dkis', u'70mm', u'710mm', u'715mm', u'718mm', u'72', u'720', u'720p', u'721mm', u'722mm', u'726mm', u'730', u'730mm', u'734mm', u'735', u'735mm', u'736', u'74', u'740mm', u'742mm', u'745mm', u'747mm', u'75', u'750', u'7500t', u'750mm', u'751mm', u'75mm', u'76', u'760', u'760mm', u'763mm', u'766mm', u'767mm', u'768', u'77', u'770', u'770mm', u'775', u'775mm', u'776mm', u'78', u'780', u'780mm', u'784mm', u'788mm', u'79', u'790mm', u'792', u'793mm', u'795mm', u'798mm', u'799mm', u'7kg', u'7mm', u'80', u'800', u'800mm', u'803mm', u'81', u'812', u'816mm', u'820mm', u'822mm', u'8240172', u'824mm', u'828', u'82mm', u'83', u'830', u'830mm', u'84', u'840mm', u'842mm', u'845mm', u'848mm', u'850', u'850mm', u'851mm', u'854', u'86', u'860', u'860mm', u'861mm', u'862', u'868mm', u'869', u'869mm', u'86mm', u'87', u'870mm', u'8713bx', u'875mm', u'880mm', u'881mm', u'884mm', u'885mm', u'888mm', u'89', u'890', u'890mm', u'892mm', u'894mm', u'895mm', u'896mm', u'897mm', u'898mm', u'899mm', u'8kg', u'8mm', u'8x', u'90', u'900', u'900mm', u'901mm', u'903mm', u'904mm', u'908mm', u'91', u'910mm', u'912mm', u'914mm', u'915mm', u'9166p', u'918', u'920', u'920mm', u'925mm', u'93', u'930mm', u'9330cdw', u'933mm', u'935mm', u'938mm', u'94', u'940mm', u'946mm', u'950', u'950mm', u'955mm', u'95mm', u'96', u'960', u'9600', u'97', u'9713ax', u'975mm', u'98', u'980', u'984mm', u'985', u'98mm', u'99', u'990mm', u'992mm', u'995mm', u'998mm', u'9kg', u'9mm', u'9s1', u'9sp', u'a1', u'a1830', u'a40bw', u'a5', u'a7', u'a7f65a', u'a8', u'a9t80a', u'a9u22a', u'aac', u'abey', u'ability', u'about', u'ac', u'accent', u'accentuate', u'access', u'accessing', u'accessories', u'accessory', u'accidently', u'accommodate', u'accomplishing', u'according', u'accounts', u'accuracy', u'accurate', u'accurately', u'acer', u'achieve', u'achieving', u'acknowledge', u'acknowledged', u'acquaintances', u'action', u'activated', u'actually', u'ad30wa', u'adapter', u'adaptor', u'add', u'added', u'addictive', u'adding', u'addition', u'additional', u'address', u'adjust', u'adjustable', u'adjustments', u'admire', u'admired', u'adopt', u'advantage', u'adventure', u'adventurers', u'ae1530', u'ae2160', u'affordably', u'after', u'afternoon', u'again', u'against', u'aging', u'agreements', u'ahead', u'air', u'airflow', u'al', u'albums', u'alcatel', u'alerts', u'ali', u'alike', u'alive', u'alkaline', u'all', u'allergens', u'allowing', u'allows', u'almost', u'along', u'already', u'also', u'altech', u'alternate', u'alternative', u'altise', u'aluminium', u'always', u'am', u'am6119', u'am7221', u'amazed', u'amazing', u'amount', u'amounts', u'ample', u'amusement', u'an', u'and', u'android', u'angle', u'animal', u'animation', u'annoyance', u'annoying', u'annual', u'another', u'answer', u'answering', u'antenna', u'anti', u'any', u'anyone', u'anything', u'anytime', u'anywhere', u'appeals', u'appearance', u'apple', u'apples', u'appliance', u'application', u'applications', u'apply', u'appreciate', u'appreciated', u'approach', u'apps', u'aquaport', u'ar12fssscwk1', u'ar24fssscwk1', u'ar30fsssbwk1', u'archaic', u'are', u'area', u'areas', u'arm', u'aroma', u'around', u'arrange', u'array', u'arrival', u'arrive', u'arriving', u'as', u'asian', u'ask', u'asko', u'assemble', u'associates', u'assortment', u'assurance', u'astg09kmca', u'astg12kmca', u'astg18kmca', u'astg30lfcc', u'at', u'atom', u'attachment', u'attachments', u'attention', u'attracted', u'attractiveness', u'au', u'auckland', u'audio', u'aus', u'australia', u'auto', u'autoclean', u'autofocus', u'automatic', u'automatically', u'auxiliary', u'available', u'avard', u'avc', u'avoid', u'away', u'awhile', u'axial', u'az100b', u'az127', u'azure', u'b2l56a', u'b70mtiss', u'baas', u'back', u'backpack', u'backup', u'bad', u'bag', u'bagels', u'bagless', u'bake', u'baked', u'bakers', u'bakery', u'baking', u'balance', u'balanced', u'bali', u'bank', u'bar', u'bare', u'bartenders', u'bas520bt', u'based', u'basic', u'bask', u'basket', u'baskets', u'bass', u'batch', u'batches', u'batter', u'batteries', u'battery', u'battling', u'baumatic', u'bbl300', u'bbl605', u'bbl605cb', u'bbq', u'bcct75n', u'bd', u'bd81gn', u'bdps1200', u'bdps3200', u'bdps5200', u'bdt160gn', u'bdve2100', u'be', u'beater', u'beats', u'beautiful', u'because', u'become', u'bedding', u'bedroom', u'beef', u'before', u'behold', u'beige', u'being', u'bejeweled', u'belkin', u'bellies', u'below', u'belt', u'bem820cb', u'bench', u'benchtop', u'beneath', u'benedict', u'benefit', u'benefits', u'berry', u'bes920cb', u'besides', u'best', u'better', u'between', u'beurer', u'beverage', u'beverages', u'beyond', u'bff', u'bfp400', u'bfp650', u'bfp800', u'bfp800cb', u'bh7540tw', u'bh97bs', u'bhm100', u'bic603s', u'bic604t', u'big', u'bigger', u'bike', u'bill', u'bills', u'birds', u'birthday', u'biscuits', u'bissell', u'bje410', u'bjs600', u'bke495', u'bke595', u'bke825', u'bke825cb', u'bke830', u'bl370', u'black', u'blacks', u'blade', u'blades', u'blanco', u'blast', u'blasts', u'blend', u'blended', u'blender', u'blending', u'blistering', u'blm800wh', u'bloatware', u'block', u'blower', u'blown', u'blu', u'blue', u'bluetooth', u'blurry', u'bm4500', u'bmo300', u'boast', u'boasts', u'body', u'boil', u'boiling', u'book', u'books', u'bookstore', u'boost', u'boosting', u'boring', u'bosch', u'bose709m', u'bose714ptx', u'bose79x', u'bose97x', u'boss', u'both', u'bother', u'bothersome', u'bottle', u'bottlenecks', u'bought', u'bowl', u'box', u'bp240', u'bp440', u'bp540', u'bpr200', u'br625t', u'braun', u'brc200', u'brc520', u'brc600', u'brce90x', u'bread', u'breadcrumbs', u'breads', u'break', u'breakfast', u'breaking', u'breeze', u'breville', u'brew', u'brewing', u'bright', u'brighten', u'brightly', u'brightness', u'brilliant', u'bring', u'bringing', u'broadcast', u'broccoli', u'broil', u'broiling', u'broom', u'brother', u'broths', u'brown', u'browning', u'browse', u'brs602x', u'brs902x', u'bru53x', u'brush', u'brushed', u'bsb530', u'bsc500', u'bsdo69', u'bsg1974', u'bsg220', u'bsg520', u'bsg540', u'bsi', u'bsk200c', u'bso65', u'bso69', u'bspo610', u'bt2600', u'bt5350', u'bta250', u'bta425', u'bta825cb', u'bta845', u'bta845cb', u'bts100', u'bts200', u'btt785gnk', u'bu', u'budget', u'buffs', u'build', u'built', u'bulk', u'bumping', u'bunnies', u'buns', u'burger', u'burgers', u'burn', u'burned', u'burner', u'burners', u'burning', u'burns', u'burst', u'bursts', u'bush', u'business', u'busy', u'but', u'butter', u'button', u'buttons', u'buy', u'bv21r050w', u'bwt740gl', u'by', u'by552xsau', u'by552xwau', u'c3520', u'c3z97a2', u'c406s', u'c460fw', u'c6gmxa8', u'c6gvxa8', u'c8031e', u'c901h', u'c9gmxa', u'cabinets', u'cable', u'cables', u'cadmium', u'caffe', u'cake', u'cakes', u'calc', u'calcium', u'calibre', u'call', u'caller', u'callers', u'calling', u'calls', u'calm', u'camcorder', u'came', u'camera', u'cameras', u'can', u'cancel', u'cancelling', u'canon', u'canopy', u'capabilities', u'capability', u'capacity', u'captivating', u'capture', u'captures', u'capturing', u'car', u'carabiner', u'card', u'cards', u'care', u'careless', u'cares', u'carpet', u'carpets', u'carrier', u'carrots', u'carry', u'carrying', u'cartridge', u'cartridges', u'carve', u'case', u'cash', u'casseroles', u'cast', u'catch', u'caught', u'caused', u'cay', u'cc4520', u'cc4840', u'ccd', u'cd', u'cd997s', u'cds', u'ce604cbx1', u'ceiling', u'celery', u'cellar', u'central', u'centre', u'centrifugal', u'ceramic', u'cf', u'cfe532wa', u'cfe742', u'cfg503wang', u'cfg504sang', u'cfg517salp', u'cfg517sang', u'cfg517walp', u'cfg517wang', u'cfm641', u'cg302dnggb1', u'cg604cwcx1', u'cg604wxc', u'cg604wxffc', u'cg903dnggb1', u'cg905dnggb1', u'cg905dwfcx1', u'cg905wxffc', u'cgg604wffc', u'cgg705wffc', u'cgg905wffc', u'ch561wa', u'ch562wa', u'ch563wa', u'ch564wa', u'challenging', u'champions', u'change', u'changes', u'channels', u'charcoal', u'charge', u'charger', u'charging', u'check', u'cheeses', u'chef', u'chefs', u'chest', u'chicken', u'child', u'children', u'chilled', u'chiller', u'chilly', u'chimney', u'choice', u'choose', u'choosing', u'chop', u'chopping', u'chops', u'chore', u'chores', u'chosen', u'chrome', u'chunking', u'chute', u'ci604dtb2', u'ci905dtb2', u'cinematic', u'cinematographers', u'cir574x', u'cir575x', u'cir597x', u'cir60x', u'circulate', u'cities', u'citrus', u'cl', u'cl641', u'cl641xl', u'cl646', u'cl646xl', u'clarity', u'classical', u'clean', u'cleaner', u'cleaners', u'cleaning', u'cleanup', u'clear', u'clearer', u'clearing', u'clearly', u'cli651bk', u'cli651c', u'cli651gy', u'cli651m', u'cli651y', u'climate', u'clips', u'close', u'closer', u'cloth', u'clothes', u'clothespins', u'clothing', u'cloud', u'cls12200', u'cls12250', u'cls12251', u'cls12253', u'cls12751', u'cls13150b', u'cls13150r', u'cls13551', u'clt', u'club', u'clutter', u'cm', u'cm4340p', u'cmos', u'cn045aa', u'cn046aa', u'cn047aa', u'cn048aa', u'co4620', u'coach', u'coaxial', u'cobalt', u'cocoa', u'coffee', u'coins', u'cold', u'colleagues', u'collect', u'collected', u'collection', u'collector', u'colour', u'colours', u'combination', u'combine', u'come', u'comes', u'comfort', u'comfortably', u'comforters', u'communicate', u'communication', u'commute', u'commuters', u'compact', u'company', u'compartments', u'compatibility', u'compatible', u'compelling', u'compensate', u'complaining', u'complement', u'complete', u'completes', u'completing', u'complex', u'complicated', u'components', u'computer', u'concealed', u'concentrate', u'condensation', u'condenser', u'condition', u'conditioner', u'conditions', u'conduct', u'conference', u'confidence', u'confident', u'confidently', u'configure', u'confined', u'confronting', u'confusion', u'congested', u'connect', u'connected', u'connecting', u'connection', u'connections', u'connectivity', u'connoisseurs', u'conquer', u'conserve', u'conserving', u'considerable', u'consistency', u'consistent', u'consistently', u'console', u'consolidate', u'constantly', u'consume', u'consumption', u'contact', u'contacts', u'containers', u'content', u'contents', u'continue', u'continuous', u'contractors', u'contrast', u'contributing', u'control', u'controls', u'convair', u'convenience', u'convenient', u'conveniently', u'conventional', u'converging', u'conversationalists', u'conversations', u'converse', u'cook', u'cooked', u'cooker', u'cookie', u'cooking', u'cooks', u'cooktop', u'cookware', u'cool', u'cooler', u'cooling', u'copier', u'copies', u'copious', u'copy', u'cord', u'cordless', u'core', u'corners', u'correctly', u'cortex', u'costly', u'costs', u'cotton', u'cottons', u'couch', u'could', u'counter', u'counting', u'countless', u'course', u'cozy', u'craftsmanship', u'cranberry', u'crank', u'cranky', u'crannies', u'craze', u'crd7000', u'cream', u'crease', u'create', u'crevice', u'crevices', u'crisp', u'crispy', u'critical', u'crop', u'cropping', u'croquettes', u'crowd', u'crumb', u'crumbs', u'crumpets', u'crushing', u'crystal', u'cs', u'ctj4003b', u'cto4003r', u'cto4003vaz', u'cto4003vbg', u'ctz4003bg', u'ctz4003bk', u'cu', u'cube', u'cucumber', u'cuf54', u'culinary', u'cup', u'cupcakes', u'cups', u'current', u'customize', u'cut', u'cutlery', u'cutting', u'cx215blue', u'cx300ii', u'cy4000', u'cy7018', u'cycle', u'cycles', u'cyclonic', u'd111s', u'd30', u'd530', u'd5424ss', u'd5434wh', u'd5457wh', u'd5532fi', u'd5544fi', u'd5544fixxl', u'd5644ss', u'd5644ssxxl', u'd5894fixxl', u'd5894ssxxl', u'd730sl', u'daily', u'dainty', u'dairy', u'damage', u'damaged', u'damp', u'dander', u'dark', u'dash', u'data', u'date', u'dated', u'daughter', u'day', u'days', u'db', u'dc', u'dc7573', u'dch6031', u'dcor', u'dcp', u'dcpf40', u'dd60dcx7', u'dd60ddfx7', u'dd90sdftx2', u'de', u'de302gb', u'de302hb', u'de30wgb', u'de40f56a2', u'de50f56a2', u'de50f56e1', u'de6010mt', u'de605ms', u'de605mw', u'de6060md', u'de608artb', u'de608m', u'de609mp', u'de60um', u'de908m', u'deal', u'dealing', u'debeta60', u'debeta90', u'debris', u'dech60s', u'dech60w', u'decide', u'decker', u'declutter', u'dect1015', u'dect1635', u'dect31151', u'dect31351', u'dedw645s', u'dedw645si', u'dedw645w', u'deep', u'def608gw', u'def905e', u'def905gw', u'defaulting', u'definition', u'defr60s', u'defrost', u'defrosting', u'degama90', u'degh60', u'degh60bg', u'degh60st', u'degh60wt', u'degh70bg', u'degh70w', u'degh90bg', u'degh90wf', u'degree', u'degrees', u'deibwc22b', u'deind603', u'deind604', u'deind804', u'deisola90', u'deisolux90', u'delayed', u'delicate', u'delicates', u'delicious', u'delight', u'delighted', u'delightful', u'delighting', u'delights', u'delivered', u'demanding', u'demands', u'dense', u'dente', u'denying', u'departure', u'dependability', u'depending', u'depriving', u'depth', u'derive', u'deriving', u'deserved', u'design', u'designed', u'designers', u'desire', u'desired', u'desires', u'desk', u'desktop', u'detail', u'detailed', u'details', u'detf115', u'detf121', u'dethalasa90b', u'detours', u'device', u'devices', u'devour', u'df4500', u'dhi955fau', u'dhl555bau', u'di664mvib3', u'di76ggesib3', u'di865mvi3', u'di965mvi3', u'dial', u'dialed', u'diameter', u'dies', u'difference', u'different', u'difficult', u'difficulty', u'digging', u'digital', u'digitize', u'dilemmas', u'dim', u'dimc10rc', u'dimc12rc', u'dimc15rc', u'dimension', u'dimplex', u'dings', u'dining', u'dinner', u'dinnerware', u'dips', u'directions', u'directly', u'dirt', u'dirty', u'disc', u'discern', u'dish', u'dishes', u'dishwasher', u'dishwashers', u'dislike', u'dispense', u'dispensed', u'dispenser', u'display', u'disregarding', u'distance', u'distant', u'distraction', u'distractions', u'disturbing', u'diversion', u'divx', u'diyers', u'dmp', u'dmr', u'do', u'docking', u'doctor', u'document', u'documents', u'dodge', u'doing', u'dolby', u'dollars', u'don', u'done', u'door', u'doors', u'dosca36x', u'dospa38x', u'double', u'dough', u'down', u'download', u'downloaded', u'downloads', u'dpi', u'dr', u'drag', u'draining', u'dramatic', u'dramatically', u'drapes', u'drawers', u'drawn', u'dreaming', u'dreams', u'dress', u'dresses', u'drink', u'drinks', u'drip', u'drive', u'drives', u'driving', u'drum', u'dry', u'dryer', u'drying', u'ds3131sp', u'ds3170', u'dslr', u'dtch60b', u'dtch80b', u'dts', u'dual', u'ducted', u'due', u'dull', u'dumpling', u'duration', u'durilium', u'during', u'dust', u'dusting', u'dusty', u'duty', u'dv1810el', u'dvd', u'dvds', u'dvi', u'dvp1012', u'dvp383t', u'dvpsr320', u'dvpsr760hpb', u'dw', u'dwa315b', u'dwa315w', u'dwau314x1', u'dwifabne', u'dwifabr', u'dww09w651a', u'dyc', u'dying', u'dyn', u'dyson', u'e12pkr', u'e18pkr', u'e249trw', u'e381trt3', u'e388lxfd', u'e3e02a', u'e402bre4', u'e440trx3', u'e442bre4', u'e442brx4', u'e521trt3', u'e522bre4', u'e522brx4', u'e522brxfd4', u'e522brxfdu4', u'e727', u'e737', u'e9pkr', u'each', u'ear', u'earth', u'ease', u'easier', u'easily', u'easy', u'eat', u'ebm4300sd', u'ebm5100sd', u'ebr7804s', u'ecam22110sb', u'ecam23460s', u'ecam45760b', u'economic', u'economical', u'edc2086pdw', u'edge', u'edges', u'edh3284pdw', u'edp2074pdw', u'educational', u'edv5051', u'edv6051', u'eek7804s', u'effect', u'effective', u'effectively', u'effects', u'efficiency', u'efficient', u'efficiently', u'effortless', u'effortlessly', u'efp6500x', u'efp9500x', u'eggs', u'ehc617w', u'ehc944ba', u'ehe5107sb', u'ehe5167sb', u'ehg643ba', u'ehg755sa', u'ehg953ba', u'ehg953sa', u'ehi645ba', u'ehi935ba', u'eight', u'elbow', u'electric', u'electrical', u'electricity', u'electrolux', u'electronic', u'electronics', u'elegant', u'element', u'eliminate', u'eliminating', u'elliptical', u'email', u'embrace', u'emergency', u'emf61mvi', u'emilia', u'emptying', u'enable', u'enables', u'enamel', u'encore', u'end', u'endless', u'enduring', u'energy', u'engage', u'engineering', u'english', u'enhance', u'enhancing', u'enjoy', u'enjoying', u'enjoyment', u'enlarge', u'enlargements', u'enough', u'ensure', u'ensuring', u'entertained', u'entertaining', u'entertainment', u'enthusiasts', u'entire', u'entries', u'environment', u'environmentally', u'environments', u'envying', u'eo400', u'eoc627s', u'eoc627w', u'episode', u'episodes', u'epson', u'equal', u'equally', u'equipment', u'equipped', u'erce9025sa', u'escape', u'esf6700rox', u'espresso', u'establish', u'estm6400', u'etam36365m', u'ethernet', u'euromaid', u'evaporative', u'eve611sa', u'eve613sa', u'eve616ba', u'eve623sa', u'eve633sa', u'even', u'evenly', u'evep611sb', u'evep613sb', u'evep616bb', u'ever', u'every', u'everyday', u'everyone', u'everything', u'ew50', u'ew60', u'ewf12832', u'ewf14742', u'ewf14912', u'eww14912', u'exact', u'exactly', u'exc627s', u'excellent', u'excitement', u'exciting', u'excursion', u'execute', u'exmor', u'expand', u'expandable', u'expanding', u'expansion', u'expansive', u'expectations', u'expenses', u'expensive', u'experience', u'experiences', u'experiencing', u'exploit', u'explore', u'extra', u'extract', u'extraction', u'extraneous', u'extreme', u'extremely', u'exynos', u'eye', u'eyes', u'eyestrain', u'ezyflix', u'f28311', u'f5100', u'f54cw', u'f54ew', u'f54gw', u'f8n657', u'fa7450', u'fa7500', u'fabulous', u'facing', u'facsimiles', u'fact', u'fail', u'failure', u'fair', u'fajitas', u'families', u'family', u'famous', u'fan', u'fans', u'far', u'fare', u'farm', u'farther', u'fashion', u'fashioned', u'fast', u'faster', u'fastest', u'fatigue', u'favorite', u'favourite', u'fax', u'faxes', u'fc032134wh', u'fc130128bc', u'fc132128rwh', u'fc182124rbc', u'fc182124rwh', u'fd9085fg', u'feast', u'feather', u'feature', u'features', u'feed', u'feel', u'feeling', u'feet', u'fellow', u'fest', u'few', u'fewer', u'fg520w', u'fi', u'figuring', u'fill', u'filling', u'film', u'filming', u'films', u'filter', u'filters', u'financials', u'find', u'finding', u'fine', u'finger', u'fingers', u'fingertips', u'finish', u'finished', u'fire', u'fires', u'first', u'fish', u'fisher', u'fishing', u'fit', u'fits', u'fitting', u'five', u'fixed', u'flac', u'flame', u'flash', u'flasks', u'flat', u'flavour', u'flexi', u'flexibility', u'flexible', u'flicker', u'flicks', u'flight', u'flip', u'flipping', u'flood', u'floor', u'floors', u'flow', u'fluffy', u'fluid', u'fluids', u'fly', u'fm', u'focal', u'focus', u'focused', u'foldable', u'folding', u'followers', u'food', u'foods', u'foodstuffs', u'football', u'for', u'forced', u'forcing', u'foreign', u'foreman', u'forget', u'forgoing', u'forgotten', u'format', u'formats', u'forms', u'forth', u'four', u'fp1200', u'fp735', u'fpm270', u'fpm810', u'fps', u'fr4068', u'fragile', u'frame', u'frames', u'free', u'freezer', u'freezing', u'french', u'frequently', u'fresh', u'freshly', u'freshness', u'fridge', u'fried', u'friendly', u'friends', u'fries', u'from', u'front', u'frost', u'frother', u'frozen', u'fruit', u'fruits', u'frustrated', u'frustrating', u'fry', u'fryer', u'frying', u'ft25', u'ft5', u'fujitsu', u'fulfill', u'fulfilling', u'full', u'fumbling', u'fun', u'function', u'functional', u'functionality', u'functions', u'fur', u'further', u'fuss', u'future', u'fv3456', u'fv5325', u'fv5335', u'fv9920', u'fz7062', u'g3', u'g4b69aa', u'galaxy', u'game', u'games', u'gaming', u'garage', u'garden', u'garments', u'garmin', u'gas', u'gases', u'gaze', u'gb', u'gc', u'gc4412', u'gc4512', u'gc4912', u'gc702', u'gear', u'gegfs60', u'generate', u'geographical', u'george', u'get', u'gets', u'getting', u'gf3tsblk', u'gfbl300', u'ghc617s', u'ghosting', u'ghr16s', u'ghs607w', u'ghz', u'gift', u'give', u'givers', u'giving', u'gl301b', u'gl4b', u'gl4bspecial', u'gl4ss', u'gl603b', u'gl704bz', u'glass', u'glide', u'global', u'gluten', u'gn', u'go', u'goal', u'goals', u'goes', u'going', u'gold', u'golden', u'goldline', u'gone', u'good', u'goodbye', u'goodness', u'goods', u'gor476sng', u'gorgeous', u'got', u'gourmets', u'gps', u'gr', u'gr18870au', u'gr6450', u'gr8210', u'grab', u'grainy', u'grandparents', u'granny', u'grapefruit', u'graphics', u'grate', u'grating', u'gray', u'grease', u'great', u'greater', u'greatest', u'green', u'grey', u'grid', u'grids', u'grill', u'grilling', u'grim', u'grime', u'grind', u'grinder', u'groceries', u'grocery', u'groove', u'grooving', u'ground', u'group', u'grow', u'grp1080au', u'gteos60', u'gtkx1bt', u'guard', u'guarding', u'guess', u'guest', u'guests', u'guidance', u'guilt', u'guilty', u'gva', u'gva23df', u'gva300', u'gva30bfc', u'gva30df', u'gva320', u'gva40pfp', u'gvabs09', u'gvadf40', u'gvadled32c', u'gvadvd7', u'gvadw60w', u'gvahv40b', u'gvahv46', u'gvamw25ss', u'gvasp4', u'gz', u'h160s', u'h215x', u'h220x', u'h510x', u'h550', u'h5500', u'h551', u'h6500', u'h6506', u'h6550wm', u'h751', u'h7750wm', u'had', u'haier', u'hair', u'half', u'halogen', u'hand', u'handheld', u'handle', u'hands', u'handset', u'handy', u'hankering', u'happen', u'happening', u'happy', u'hard', u'hardcopy', u'hardwood', u'harness', u'has', u'hassle', u'hate', u'hating', u'have', u'having', u'hba13b253a', u'hba23b151a', u'hba63b250a', u'hba63s451a', u'hbg43s450a', u'hbg73s550a', u'hbm43b250a', u'hbm43s550a', u'hc60pchtx2', u'hc90cgx1', u'hd', u'hd201', u'hd201s', u'hd429s', u'hd8752', u'hd9220', u'hdmi', u'hdr', u'hdr9650ts', u'hdrcx240', u'hdrpj240', u'hdrpj540', u'headaches', u'headphones', u'healthful', u'healthy', u'hear', u'heard', u'hearing', u'heart', u'heat', u'heated', u'heater', u'heating', u'heavy', u'hectic', u'height', u'helmet', u'help', u'helps', u'hepa', u'herbs', u'herd', u'here', u'hewlett', u'high', u'highest', u'highlights', u'highly', u'hinged', u'hip', u'hisense', u'hit', u'hits', u'hitting', u'hl', u'hms', u'hobbs', u'hold', u'holder', u'holiday', u'holidays', u'home', u'homemade', u'homemakers', u'homey', u'honey', u'hood', u'hook', u'hooks', u'hoover', u'hop', u'hose', u'host', u'hosting', u'hot', u'hottest', u'hour', u'hours', u'house', u'household', u'housing', u'how', u'however', u'hp2500', u'hp5520', u'hp60idchx2', u'hp90idchx2', u'hr', u'hr6af243', u'hr6bf121', u'hr6bf121s', u'hr6bf47', u'hr6cf146', u'hr6cf206', u'hr6cf307', u'hr6tff222', u'hr6tff342b', u'hr6tff437', u'hr6tff437sd', u'hr6tff527', u'hr6tff527sd', u'hr6vff177a', u'hr6wc29', u'hr7776', u'hr945t', u'hs60csx3', u'hs950', u'ht', u'htc', u'htct370', u'htl2163', u'htl9100', u'htm77', u'huge', u'humax', u'humid', u'humidity', u'hungry', u'hurry', u'hw', u'hw220glk', u'hwfr6510', u'hwm75tlu', u'hwmp65', u'hygienic', u'hz', u'ice', u'icf304s', u'iconia', u'id', u'idea', u'ideal', u'identical', u'if', u'ignition', u'ignoring', u'ii', u'iis', u'ilce3500j', u'ilce5000lb', u'ill', u'illuminate', u'image', u'imagery', u'images', u'immerse', u'immersing', u'important', u'impress', u'impressed', u'impressing', u'improve', u'improving', u'impurities', u'in', u'inability', u'incandescent', u'inch', u'include', u'includes', u'increase', u'increased', u'increasing', u'indicators', u'individual', u'indoor', u'indoors', u'induction', u'information', u'ingredients', u'injuries', u'inkjet', u'inlet', u'inner', u'input', u'inputs', u'ins', u'insert', u'inserts', u'inside', u'inspiration', u'instacube', u'install', u'installation', u'installed', u'instant', u'instantly', u'instead', u'integrated', u'intel', u'intensive', u'intently', u'interact', u'interior', u'internet', u'intersections', u'interview', u'into', u'intrinsic', u'intuitive', u'intuitively', u'invading', u'inverter', u'invest', u'investment', u'invite', u'ion', u'ios', u'ipad', u'iphone', u'ipm', u'ipod', u'iron', u'ironing', u'irritants', u'is', u'is4110s1', u'is4110s2', u'is4110sp', u'is4140s1', u'is4140sp', u'isight', u'island', u'isn', u'iso', u'issue', u'issues', u'it', u'items', u'its', u'j172w', u'j6920dw', u'j870dw', u'jabra', u'jack', u'jaffles', u'jar', u'jazz', u'jbcharge2blk', u'jbcharge2red', u'jbflipiiblkas', u'jbl', u'je2700', u'je5600', u'je9000', u'jeans', u'jm3250', u'jm6600', u'job', u'jobs', u'join', u'journey', u'jpeg', u'ju', u'juice', u'juicer', u'juices', u'juicing', u'julienne', u'jump', u'junctions', u'just', u'jvc', u'k181x90', u'k406s', u'kaiser', u'kak36', u'kambrook', u'karcher', u'kbj2001b', u'kbj2001w', u'kbl120frd', u'kbl210', u'kbl600', u'kbo2001r', u'kbo2001vaz', u'kbo2001vbg', u'kbo2001vgr', u'kbz2001gy', u'kbz2001w', u'kd65x9000b', u'kd79x9000b', u'kdf460', u'kdl32w700b', u'kdl40w600b', u'kdl48w600b', u'kdl55w800b', u'kdl60w850b', u'ke2350', u'ke3560', u'ke6300', u'ke7110', u'keep', u'keeping', u'kelvinator', u'kenwood', u'kettle', u'keyring', u'kfa213', u'kfa715', u'kfa835', u'kfp400', u'kg', u'ki450', u'kick', u'kid', u'kids', u'kill', u'kind', u'kindle', u'kinds', u'kinetix', u'kiss', u'kit', u'kitchen', u'kitchenware', u'kj12', u'km280', u'knead', u'kneading', u'knob', u'know', u'knowing', u'kpr600', u'krc5', u'ks19', u'ksb100frd', u'ksb7', u'ksc120', u'ksk210', u'ksk210f', u'ksv26hre', u'ksv70hre', u'kt450f', u'kt50', u'kt60', u'ktm4200wb', u'kw', u'kwh15cme', u'kwh26hre', u'kx', u'l218asl', u'l2340dw', u'l23f3380', u'l28b2500', u'l32b2600', u'l40b2800f', u'l50s5600fs', u'l55b3700f', u'l65e5510fds', u'l730sl', u'labels', u'labour', u'lack', u'lacking', u'lamb', u'lamenting', u'land', u'landline', u'landscapes', u'lane', u'lanes', u'lanyard', u'laptop', u'large', u'larger', u'laser', u'last', u'latest', u'latte', u'launder', u'laundered', u'laundering', u'laundromat', u'laundry', u'lay', u'layer', u'layout', u'lc131bk', u'lc131c', u'lc131m', u'lc133', u'lc133c', u'lc133m', u'lc133y', u'lc139xlbk', u'lc5000', u'lc6250', u'lc70le650x', u'lc9000', u'lcd', u'ld1482s4', u'ld1482w4', u'ld1483t4', u'ld1484t4', u'leap', u'leave', u'leaving', u'led', u'left', u'leftovers', u'leg', u'legal', u'legriaminix', u'length', u'lenoxx', u'lens', u'lenses', u'less', u'let', u'lets', u'letters', u'letting', u'lev24a1fhd', u'level', u'levels', u'leverage', u'lg', u'library', u'lid', u'life', u'lifetime', u'lift', u'lifting', u'light', u'lighter', u'lighting', u'lights', u'like', u'liked', u'liking', u'limitations', u'limited', u'limiting', u'line', u'linens', u'liner', u'linger', u'lingerie', u'lining', u'link', u'list', u'listen', u'listening', u'lite', u'lithium', u'litre', u'little', u'live', u'living', u'll', u'load', u'loading', u'loads', u'loaf', u'loathing', u'loathsome', u'loaves', u'local', u'locate', u'location', u'lock', u'lof26a9s', u'lofcp750ss', u'lofsl260iss', u'logitech', u'long', u'longer', u'longhi', u'look', u'looking', u'looks', u'losing', u'loss', u'lost', u'lot', u'lots', u'loud', u'love', u'loved', u'lovers', u'low', u'lowepro', u'lower', u'lp', u'lpcm', u'ls20d300hy', u'lugging', u'lukewarm', u'lumia', u'lump', u'lumpy', u'lunch', u'lush', u'luxacar01', u'luxacaw01', u'luxor', u'm2070fw', u'm406s', u'mac', u'machine', u'machines', u'made', u'magnitude', u'mah', u'mail', u'maintain', u'maintaining', u'maintenance', u'make', u'maker', u'making', u'man', u'manage', u'managing', u'manganese', u'manual', u'manually', u'manufactured', u'manufacturer', u'many', u'map', u'maps', u'margaritas', u'marvel', u'master', u'masticating', u'match', u'matches', u'material', u'materials', u'matter', u'maximise', u'maximum', u'mb', u'mb810', u'mc8289ur', u'mc9280xc1', u'md477zp', u'md480zp', u'md531x', u'md543x', u'md720zp', u'md788x', u'md791x', u'md827fe', u'mdrnc8b', u'mdrnc8bs', u'mdrxb400b', u'mdrxb400bs', u'mdrzx310apb', u'mdrzx310apbs', u'me6124st', u'me6124w', u'me6144st', u'me6144w', u'meal', u'meals', u'means', u'meant', u'measuring', u'meat', u'meats', u'mec', u'media', u'medium', u'meet', u'meeting', u'meets', u'mega', u'melbourne', u'melodious', u'members', u'memorable', u'memories', u'memory', u'memos', u'menu', u'menus', u'mercator', u'mercy', u'meringues', u'mesh', u'mess', u'messages', u'messes', u'metal', u'metallic', u'meters', u'method', u'mew', u'mfc', u'mg2560', u'mg3560', u'mg6660bk', u'mg7160bk', u'mhcv3', u'micro', u'microphone', u'microsd', u'microsdhc', u'microwave', u'miele', u'milc', u'milestone', u'mill', u'min', u'mini', u'minimal', u'minimise', u'minimize', u'minimizing', u'minimum', u'minute', u'minutes', u'mirror', u'misplaced', u'miss', u'missing', u'mist', u'mitsubishi', u'mix', u'mixer', u'mixes', u'mixing', u'mkv', u'ml', u'mlt', u'mm30i', u'mobile', u'mobility', u'mode', u'modern', u'modernize', u'modernizing', u'modes', u'modify', u'moisten', u'moisture', u'moment', u'moments', u'money', u'monitor', u'mood', u'more', u'morning', u'mornings', u'mos', u'most', u'motion', u'moto', u'motor', u'motorcycle', u'motorized', u'motorola', u'mount', u'mountable', u'mountain', u'mounted', u'mounting', u'move', u'movement', u'moves', u'movie', u'movies', u'moving', u'mozart', u'mp', u'mp3', u'mp3s', u'mp4', u'mpeg', u'mpo', u'ms', u'ms2540sr', u'ms3042g', u'ms3882xrsk', u'much', u'muck', u'muffins', u'multi', u'multifunction', u'multiple', u'multiprocessor', u'multitask', u'multitude', u'mundane', u'music', u'musical', u'must', u'mw513', u'mw60', u'mx320', u'mx365blue', u'mx375', u'mx476', u'mx5950', u'mx7900r', u'mx8500w', u'names', u'narrow', u'national', u'natural', u'nature', u'navigate', u'navigation', u'navman', u'nb3540', u'nb4540', u'neat', u'necessary', u'need', u'needed', u'needing', u'needs', u'neglecting', u'neighbour', u'network', u'networks', u'never', u'new', u'news', u'newsletters', u'next', u'nfc', u'nice', u'nicely', u'nickel', u'night', u'nights', u'nikon', u'nirvana', u'nn', u'no', u'noise', u'nokia', u'non', u'nooks', u'not', u'notches', u'note', u'notebook', u'novels', u'nr', u'ntsc', u'number', u'numbers', u'nw541etcsta', u'nw541etcwhi', u'nw541gtcfsta', u'nw541gtcfstalpg', u'nw541gtcsta', u'nw541gtcstalpg', u'nw601etcsta', u'nw601gtcsta', u'nw601gtcwhi', u'nw601gtcwhilpg', u'ob60b77cew3', u'ob60b77cex3', u'ob60s9dex2', u'ob60scex4', u'ob60sl11dcpx1', u'ob60sl11depx1', u'ob60sl7dew1', u'ob60sl7dex1', u'ob60sl9dex1', u'ob90s9mepx2', u'obsolete', u'obstructions', u'oc64kz', u'oc64tz', u'oc70tz', u'oc95txa', u'octa', u'od152w', u'odours', u'odw702xb', u'of', u'of991xs', u'off', u'offer', u'offering', u'office', u'often', u'og60xa', u'og63x', u'og72xa', u'og91x', u'ohv40c', u'ohv46c', u'ohvp46c', u'oi64z', u'oil', u'oils', u'ois', u'old', u'oled', u'olimpia', u'olive', u'olympus', u'omega', u'on', u'once', u'one', u'ones', u'online', u'only', u'onto', u'oo654x', u'oo664x', u'oo686x', u'oo6ax', u'oo757x', u'op8611ss', u'op8621a', u'opening', u'operate', u'operating', u'operation', u'opportunities', u'opportunity', u'optical', u'optimal', u'optimum', u'option', u'options', u'or', u'orange', u'orc97g', u'order', u'organize', u'organized', u'organizing', u'orient', u'ort60x', u'ort6wxa', u'os', u'oscar', u'oscillation', u'ot8601ss', u'other', u'others', u'out', u'outdated', u'outdoor', u'outlet', u'outmoded', u'output', u'outputs', u'outs', u'outside', u'outstanding', u'oven', u'over', u'overcome', u'overcooked', u'overheated', u'overly', u'oversized', u'overwhelmed', u'own', u'owners', u'p227fsl', u'pa9s2', u'paccn86', u'pace', u'paced', u'pack', u'packages', u'packard', u'padded', u'pages', u'pair', u'pairing', u'pale', u'palette', u'pan', u'panasonic', u'panel', u'panini', u'paninis', u'panoramic', u'pans', u'paper', u'par', u'parents', u'parking', u'parrot', u'part', u'partial', u'particles', u'parties', u'parts', u'party', u'pass', u'passe', u'past', u'paying', u'paykel', u'pb2000', u'pb7620', u'pb7630', u'pb9800', u'pbh615b9ta', u'pcd240b', u'pci815b91a', u'pcq715b90a', u'pd9030', u'pdvd830', u'pedestal', u'penta', u'pentaprism', u'people', u'perceived', u'perfect', u'perfection', u'perfectly', u'perform', u'performance', u'performing', u'periods', u'perishable', u'perishables', u'perk', u'personal', u'pesky', u'pesto', u'pet', u'pf560014', u'pf560014s', u'pg', u'pg640', u'pg640xl', u'pg645', u'pg645xl', u'pga64', u'pgi', u'pgi650bk', u'philips', u'phone', u'phonebook', u'photo', u'photographed', u'photographers', u'photographic', u'photography', u'photos', u'phr284u', u'phr395u', u'pick', u'picking', u'pics', u'picture', u'pictures', u'pie', u'pie651f17e', u'pie675n14e', u'pies', u'pil611b18e', u'pile', u'pin875n17e', u'pinch', u'pink', u'pit651f17e', u'pit851f17e', u'piu12', u'piu16', u'pixel', u'pizza', u'pkn675n14a', u'place', u'placement', u'plan', u'planet', u'plastic', u'plated', u'plates', u'platform', u'platters', u'play', u'player', u'players', u'playing', u'playlist', u'playlists', u'plays', u'pleasant', u'please', u'pleasure', u'plenty', u'plinth', u'plug', u'plugging', u'plumbing', u'plus', u'pocket', u'point', u'polk', u'pollen', u'pollutants', u'polyphonic', u'poor', u'pop', u'popcorn', u'popular', u'por663s', u'pork', u'port', u'portability', u'portable', u'portions', u'portraits', u'position', u'possession', u'possibilities', u'possible', u'potato', u'potatoes', u'pots', u'pouch', u'pounding', u'power', u'powered', u'powerful', u'powerfulness', u'ppm', u'ppq716b21a', u'pr129', u'pr192', u'practical', u'practice', u'prd6w', u'pre', u'precious', u'precise', u'precisely', u'precision', u'prefer', u'preferred', u'premium', u'prep', u'prepare', u'present', u'preservative', u'preserve', u'preset', u'presets', u'presettings', u'press', u'pressing', u'pressure', u'pretty', u'prevent', u'pride', u'print', u'printer', u'printing', u'printouts', u'pristine', u'private', u'prized', u'pro', u'probably', u'problem', u'processing', u'processor', u'produce', u'product', u'productive', u'professional', u'professionals', u'program', u'programs', u'progress', u'promising', u'proper', u'properly', u'pros', u'protect', u'protecting', u'protection', u'provide', u'providers', u'providing', u'provisions', u'publish', u'pull', u'pulping', u'pulse', u'pump', u'punch', u'purchases', u'puree', u'purified', u'purple', u'push', u'put', u'putting', u'pv1820l', u'pvr9600q', u'pwt540gl', u'px1181e', u'px30ii', u'pyrolytic', u'quad', u'quality', u'quantities', u'quarters', u'quick', u'quickflix', u'quickly', u'quietly', u'r10686', u'r2', u'r20a0w', u'r330ys', u'r330yw', u'r350yw', u'r395ys', u'r54cw', u'racing', u'rack', u'racks', u'radio', u'raincoat', u'rainy', u'range', u'rapidly', u'rasping', u'rate', u'rating', u'ratio', u'ravenous', u'ray', u'rc5600', u'rc689d', u'rca', u'rca2ai6wh', u're', u'reach', u'read', u'reader', u'reading', u'ready', u'readying', u'real', u'realistic', u'reality', u'really', u'reap', u'reaping', u'rear', u'reason', u'receive', u'recharge', u'rechargeable', u'recharging', u'recipe', u'recipes', u'recirculating', u'recognition', u'recognized', u'record', u'recorded', u'recorder', u'recording', u'red', u'reduce', u'reducing', u'reduction', u'refill', u'refined', u'reflective', u'refresh', u'refrigerator', u'regard', u'regardless', u'region', u'regular', u'regulate', u'reheat', u'reheating', u'reinforced', u'relatively', u'relatives', u'relax', u'relaxing', u'reliability', u'reliable', u'relish', u'relished', u'relishing', u'relying', u'remain', u'remedy', u'remote', u'removable', u'removal', u'remove', u'removing', u'repair', u'replace', u'replacement', u'replacing', u'report', u'reports', u'required', u'requirements', u'reservoir', u'residual', u'resize', u'resolution', u'resources', u'responding', u'response', u'responsibly', u'responsive', u'rest', u'resting', u'results', u'resumes', u'retina', u'revel', u'reveling', u'reversible', u'revitalize', u'rewarded', u'rf', u'rf522adusx4', u'rf522adw4', u'rf522adx4', u'rf610adusx4', u'rf610adx4', u'rfd602s', u'rh735t', u'rhc909', u'rhg501whi', u'rhii9ss', u'rhk32pur', u'rhk32red', u'rhk4w', u'rhmfp1', u'rhmp750', u'rice', u'rid', u'riddance', u'ride', u'rides', u'riding', u'rie3cl9ss', u'right', u'ring', u'ringtone', u'rinse', u'risotto', u'rles91ss', u'ro61ss', u'road', u'roast', u'roasted', u'roasting', u'robinhood', u'rock', u'rocking', u'role', u'roof', u'room', u'rooms', u'rotary', u'rotation', u'rotisserie', u'round', u'route', u'routes', u'row', u'rpb3cl6ss', u'rpm', u'rre634s', u'rs110', u'rs120', u'rsa2cl6ss', u'rsa2cl9ss', u'ruin', u'rumpled', u'run', u'runners', u'running', u'russell', u'rw', u'rwb3cf9ss', u'rwc3ch6ss', u'rwc3ch9ss', u'rwc3ch9wh', u'rwe3ch9ss', u'rwh3ch6ss', u'rwh3ch9ss', u'rwv3cl6g', u'rwv3cl9g', u'rx120b', u's09awn', u's12awn', u's24awn', u's4', u's5', u'sa980cxa', u'sacrificing', u'saeco', u'safe', u'safely', u'safer', u'safety', u'sale', u'sales', u'same', u'samsung', u'sandwich', u'sandwiches', u'sangean', u'satellite', u'satisfy', u'satisfying', u'sauce', u'sauces', u'saute', u'save', u'saving', u'savings', u'savory', u'savour', u'savouring', u'say', u'sb303', u'sbt30blk', u'sbt30pnk', u'sc', u'sca706x', u'sca709x', u'scale', u'scan', u'scanner', u'schedule', u'schweigen', u'scrambling', u'scraper', u'scrapes', u'scratched', u'scratches', u'screen', u'screw', u'sd', u'sdhc', u'sdxc', u'seafood', u'seamless', u'seamlessly', u'searching', u'season', u'seconds', u'secure', u'security', u'see', u'seeing', u'seekers', u'seeking', u'seen', u'select', u'selecting', u'selection', u'selections', u'send', u'sennheiser', u'sensitivity', u'sensor', u'sequence', u'series', u'serve', u'server', u'services', u'serving', u'servings', u'set', u'setdxk09zma', u'setdxk12zma', u'setdxk24zma', u'setting', u'settings', u'settle', u'settling', u'seven', u'several', u'sfa125', u'sfa130', u'sfa304x', u'sfa309x', u'sfa395x', u'sfpa125', u'sfpa130', u'sfpa140', u'sfpa390x', u'sfpa395x', u'shade', u'shakes', u'shaking', u'shaped', u'share', u'sharp', u'sharper', u'shb7150', u'shb7150s', u'shb8000bk', u'shb8000bks', u'shc5100', u'shc5100s', u'shc700x', u'shc8535', u'shc8535s', u'shd8600', u'shd8600s', u'she', u'sheet', u'sheets', u'shelf', u'shelves', u'shelving', u'shine', u'shirt', u'shirts', u'shl140', u'shl140s', u'shoot', u'shooting', u'shoppers', u'short', u'shortcuts', u'shot', u'shots', u'should', u'shoulder', u'show', u'shows', u'shq3200', u'shq4200', u'shredding', u'shu500x', u'shut', u'shutterbugs', u'shw900b', u'shw910x', u'side', u'sided', u'signature', u'signed', u'sihp263s', u'silence', u'silk', u'silky', u'silly', u'silver', u'sim', u'simmer', u'simmering', u'simple', u'simplify', u'simplifying', u'simply', u'simpson', u'simultaneously', u'single', u'sink', u'sites', u'sitting', u'situation', u'situations', u'six', u'size', u'sized', u'sizes', u'sj244vwh', u'sj308vwh', u'sjf624stsl', u'sjf676stsl', u'sjfj676vbk', u'skid', u'skills', u'skip', u'sl', u'sleek', u'sleeker', u'sleeve', u'slh260iss', u'slice', u'slicing', u'slide', u'slideshows', u'slot', u'slow', u'slower', u'sluggish', u'sm7200', u'sm7400', u'sm9000', u'small', u'smart', u'smeg', u'smell', u'smelling', u'smells', u'smoke', u'smooth', u'smoothie', u'smoothies', u'smoothing', u'sms40m12au', u'sms50e32au', u'sms63l08au', u'sms68m22au', u'sms69t18au', u'smu68m15au', u'snack', u'snacks', u'snapdragon', u'snappy', u'snapshot', u'sneak', u'sneaking', u'snhd439', u'snpx100ii', u'snpx100iis', u'so', u'soak', u'social', u'soft', u'softening', u'solari', u'soleplate', u'solid', u'solitaire', u'solution', u'solve', u'some', u'song', u'sony', u'soon', u'sorely', u'sound', u'sounded', u'sounding', u'sounds', u'soundtrack', u'soup', u'soups', u'source', u'sp', u'space', u'spaces', u'spacious', u'spaghetti', u'spare', u'sparingly', u'sparkle', u'sparkling', u'spatula', u'speak', u'speaker', u'speakerphone', u'speakers', u'special', u'specifications', u'speed', u'spend', u'spending', u'spent', u'spills', u'spillsafe', u'spin', u'splash', u'spoil', u'spoiling', u'spoilt', u'spoken', u'spoon', u'sport', u'sports', u'spray', u'spreads', u'spritz', u'spruce', u'sq', u'square', u'squeeze', u'squeezed', u'squeezing', u'sr254mw', u'sr255mls', u'sr320mls', u'sr415mls', u'sr469mls', u'sr6250', u'srf679swls', u'srf680cdls', u'srf828scls', u'srf890swls', u'srfs84c', u'srs580dhls', u'srs636scls', u'srs676gdhls', u'srsx2b', u'srsx3b', u'srsx5b', u'srtrb2', u'sse35', u'st253bqpq', u'st342wqpq', u'st641w', u'st663wqpq', u'st671s', u'st683sqpq', u'stabiliser', u'stage', u'stainless', u'stairs', u'stalks', u'standard', u'standards', u'standby', u'standing', u'star', u'starching', u'stare', u'start', u'stash', u'station', u'stations', u'stay', u'staying', u'steadier', u'steadily', u'steady', u'steaks', u'steam', u'steamed', u'steamer', u'steamglide', u'steaming', u'steel', u'steer', u'stereo', u'stew', u'stick', u'stickers', u'still', u'stills', u'stir', u'stm', u'stock', u'stocked', u'stocking', u'stockpile', u'stockpiling', u'stone', u'stop', u'storage', u'store', u'storing', u'stove', u'stovetop', u'straining', u'strap', u'stream', u'streamer', u'streaming', u'streamlined', u'street', u'streets', u'strength', u'stress', u'stretch', u'stretching', u'strive', u'strong', u'struggling', u'stubborn', u'stuck', u'students', u'studio', u'stuffy', u'stunning', u'style', u'stylish', u'sub', u'subject', u'subjects', u'subwoofer', u'successful', u'successfully', u'succumbing', u'suction', u'suffering', u'suit', u'suits', u'summer', u'summertime', u'sun', u'sunbeam', u'sunday', u'sunny', u'supply', u'supplying', u'supports', u'sure', u'surface', u'surfaces', u'surround', u'swamped', u'sweating', u'sweeping', u'sweet', u'swf10732', u'swiftly', u'swim', u'switch', u'swt5542', u'swt6042', u'swt8542', u'swt9542', u'sx400is', u'sx520hs', u'sx600hs', u'sync', u'system', u'systems', u't754chp', u't784chp', u't794cfi', u'ta4200', u'ta4400', u'ta60ss', u'ta6440', u'ta90ss', u'tab', u'table', u'tablet', u'tabs', u'tack', u'tackle', u'take', u'taking', u'talk', u'talked', u'talking', u'tallboy', u'tangential', u'tangled', u'tangy', u'tank', u'tap', u'target', u'tasking', u'tasks', u'taste', u'tb', u'tcl', u'td', u'tea', u'teac', u'team', u'teatime', u'tech', u'techies', u'technika', u'techniques', u'technology', u'tedious', u'tefal', u'teflon', u'teg64ua', u'teg75u', u'teg85u', u'teg95hua', u'teg95ua', u'telephone', u'telescopic', u'television', u'telling', u'telstra', u'temperature', u'temperatures', u'tempered', u'tempura', u'tend', u'terribly', u'text', u'tft', u'tg1611alh', u'tg2723alm', u'tg6821alb', u'tg6822alb', u'tgc6gwss', u'tgc7gwss', u'tgc7ind', u'tgc9glwss', u'tgdo84tbs', u'tgg64u', u'tgg75u', u'tgo610ftbs', u'tgo65bs', u'tgo68tbs', u'tgo910ftbs', u'tgslh850ss', u'tgso618ftbs', u'th', u'than', u'thanks', u'that', u'thaw', u'thawing', u'the', u'theatre', u'their', u'them', u'theme', u'there', u'thermometer', u'thermostat', u'these', u'they', u'thick', u'thing', u'this', u'those', u'three', u'thrill', u'thrilling', u'through', u'throughout', u'tideous', u'tilt', u'time', u'timely', u'timer', u'timers', u'times', u'timesaving', u'tiniest', u'tionicglide', u'tip', u'tired', u'titanium', u'titles', u'tn', u'to', u'toast', u'toasted', u'toaster', u'toasting', u'toasty', u'together', u'tom', u'tomorrow', u'ton', u'tones', u'tonight', u'tons', u'too', u'tool', u'top', u'toshiba', u'toss', u'total', u'tote', u'touch', u'touching', u'touchscreen', u'tough', u'toughest', u'towels', u'tower', u'towns', u'track', u'tracked', u'traditional', u'traffic', u'transact', u'transfer', u'transferring', u'transmit', u'transport', u'transtherm', u'travels', u'tray', u'trays', u'treasure', u'treasured', u'treat', u'treble', u'treks', u'tremendous', u'trend', u'tricky', u'tricot', u'trip', u'triple', u'trips', u'trivets', u'trouble', u'truck', u'true', u'truehd', u'trust', u'try', u'trying', u'ts715a', u'ttm030p', u'tumble', u'tune', u'tuner', u'tunes', u'tuning', u'turbo', u'turn', u'turning', u'turns', u'turntable', u'tv', u'twin', u'two', u'type', u'types', u'tz40', u'tz60', u'u1850', u'u3407', u'u40e5691fds', u'u6011', u'u65e5691fds', u'u7820', u'ua19h4000aw', u'ua22h5000aw', u'ua28h4000aw', u'ua32h4000aw', u'ua32h5500aw', u'ua32h6400aw', u'ua40h5000aw', u'ua40h6400aw', u'ua48h5000aw', u'ua48h6400aw', u'ua55h6400aw', u'ua55hu7200w', u'ua55hu8500w', u'ua55hu9000w', u'ua60h6400aw', u'ua60h7000aw', u'ua65h6400aw', u'ua65hu7200w', u'ua65hu8500w', u'ua75h6400aw', u'ua75h7000aw', u'ua78hu9000w', u'uec', u'uef54', u'uhs', u'ul', u'ultimate', u'ultra', u'um', u'um1170', u'unattended', u'under', u'undermount', u'unequipped', u'unfamiliar', u'unfold', u'unhappily', u'unhealthful', u'unhealthy', u'uniden', u'uninspired', u'unintentionally', u'uninterrrupted', u'unit', u'universal', u'unlocked', u'unnecessary', u'unobtrusively', u'unpleasant', u'unprotected', u'unsightly', u'unusually', u'unwanted', u'up', u'updates', u'upgrade', u'upgraded', u'upgrades', u'upgrading', u'upholstery', u'upkeep', u'upload', u'uploading', u'upper', u'upright', u'ups', u'usage', u'usb', u'use', u'useful', u'user', u'using', u'utilize', u'v110sges3', u'v12btcc', u'v190sg2ebk', u'v204042ba000', u'v207020sa010', u'v233031', u'v233086', u'v40cpf', u'vacation', u'vacuum', u'vain', u'valances', u'valuable', u'value', u'vanguard', u'vapours', u'vari', u'variable', u'varieties', u'variety', u'various', u'vast', u'vax', u'vba430aa', u'vbk330za', u'vbk370za', u'vc1352', u'vcd', u'vcdf30', u'vcp7p2400', u'vczp1600', u'vczph1600', u've', u'vegetables', u'veggies', u'venting', u'verify', u'versatile', u'version', u'vertical', u'very', u'vga', u'vggcer64', u'vggh64ss', u'vhd', u'via', u'viali', u'vibrant', u'video', u'videos', u'view', u'viewed', u'viewfinder', u'viewing', u'vintec', u'viral', u'virtually', u'vision', u'visitors', u'vista', u'visually', u'visuals', u'vital', u'vitamins', u'vivid', u'vjp', u'voice', u'voicemail', u'volta', u'volume', u'vtrs', u'w450upl', u'w600gsl', u'w6444', u'w6564', u'w6864', u'w6984fi', u'w810', u'w830', u'w8844xleco', u'wa1068g1', u'wa65f5s2urw', u'wa70t60gw1', u'wa75f5s6dra', u'wa80t65gw1', u'waffles', u'waiting', u'walk', u'wall', u'walls', u'want', u'warm', u'warmer', u'warmest', u'warming', u'warmth', u'warranty', u'warriors', u'wash', u'washable', u'washed', u'washer', u'washing', u'waste', u'wasting', u'watch', u'watching', u'water', u'waterproof', u'watt', u'watts', u'wav', u'waves', u'way', u'way32840au', u'wb350f', u'wb50f', u'wbb3700pa', u'wbe5100sc', u'wbm4300wb', u'wcm1500wc', u'wcm2100wc', u'wcm3200wc', u'wcm7000wc', u'wd10f7s7srp', u'wd12021d6', u'wd14022d6', u'wd14024d6', u'wd14071sd6', u'wd3330m', u'weathering', u'web', u'weekend', u'weigh', u'weighing', u'weight', u'well', u'wels', u'were', u'westinghouse', u'wet', u'wf3620', u'wfe914sa', u'wfm0900wc', u'wfm1800wd', u'wfm1810wc', u'wfm3000wb', u'wfm3600sb', u'wh', u'wh7560j1', u'wh7560p1', u'wh8560j1', u'wh8560p1', u'what', u'whe5100sa', u'wheat', u'when', u'whenever', u'where', u'wherever', u'whether', u'which', u'while', u'whip', u'whipping', u'whisk', u'white', u'whites', u'who', u'whole', u'why', u'wi', u'wide', u'width', u'wild', u'will', u'wim1200sc', u'wim1200wc', u'window', u'windows', u'wine', u'winter', u'wipe', u'wire', u'wireless', u'wirelessly', u'wires', u'wise', u'wish', u'with', u'within', u'without', u'wl1068p1', u'wle525wa', u'wle535wa', u'wle645wa', u'wlg503walp', u'wlg503wang', u'wlg517wang', u'wm2190', u'wm2190s1', u'wm3150sp', u'wm5', u'wma', u'wmv', u'wok', u'won', u'wonder', u'wonderful', u'wooden', u'wool', u'work', u'workers', u'working', u'workout', u'works', u'workspace', u'worktop', u'world', u'worry', u'worrying', u'worst', u'worth', u'would', u'wowing', u'wrf900cs', u'wrh605is', u'wrh608is', u'wrh608iw', u'wrh908is', u'wrinkle', u'wrinkled', u'wrinkles', u'writing', u'wrj600us', u'wrj600uw', u'wrj900us', u'wrj900uw', u'wrj911us', u'wrm1300wc', u'wrm2400wd', u'wrm3700wb', u'wrm4300sb', u'wse6070sf', u'wse6100wf', u'wse6970sf', u'wse7000sf', u'wse7000wf', u'wt', u'wta74200au', u'wtb2300wc', u'wtb3400wc', u'wtb86200au', u'wte4200sb', u'wte5200sb', u'wtm4200wb', u'wtm5200wb', u'wty88700au', u'ww10h8430ew', u'ww85h7410ew', u'xa20', u'xdect81551', u'xdect81552', u'xdectr0552', u'xe', u'xp', u'xperia', u'xsa', u'xw440glk', u'xy', u'y406s', u'yamaha', u'yas', u'year', u'yelled', u'yellow', u'yet', u'yht', u'you', u'your', u'yourself', u'yourselfers', u'yv9601', u'z2', u'zap', u'zb3010', u'zb3012', u'zc500', u'ze347', u'zipper', u'zone', u'zones', u'zoom', u'zsc6930', u'zss10cp', u'zss3ipn']
        text_fields = apply_filt(str,map(extract_text,lines))       
        
     
        #Vectorizer for unigrams
        vectorizer = CountVectorizer(min_df=1,binary=True)
        #vectorizer.fit_transform(top_100_unigrams_no_stop)
        vectorizer.fit_transform(map(unicode,words))
        f_words = vectorizer.transform(text_fields).toarray()
        
     


        ########################################################################

        #Features for the model

        #Key:
        #f1-log(price) + f10-male_words + f2-gender + f15-age+ f16-popularity + f0-category np_cart
        #f9-adj + f14-kwd + f12-originality(removed causing nans) + f14-readability2 + f7-problem + f5-benefit_cnt + f8-num_features
        X = numpy.array(zip(x0,fb))

        # #baseline, price, popularity
        #X = numpy.array(zip(x0,fb,f1_log))     

        # f18-top 100 unigrams
        allfeatures = numpy.concatenate((X,f_words),axis=1)
        
        return allfeatures
        #return X


    def process_y_values(lines):
        y = apply_filt(float,map(extract_y,lines))
        Y = numpy.array([i+0.0000001 if i==0 else i for i in y])  #add small value to 0 probabilities)
        return Y


    def process_training_features(lines, baseline, words):
        """Extracts features from lines, normalizes feature vectors to have mean 0 and unit standard deviation"""

        def normalize_training_data(X):
            """normalizes training data and returns normalized X array and normalizers so they can be used for processing the test set"""
                
            normalizers=[]
            for i in range(X.shape[1]-2):   #don't need to normalize first 2 features: bias,baseline
                mean=X[:,i+2].mean()
                sd=X[:,i+2].std()
                normalizers.append([mean,sd])
                X[:,i+2] = stats.zscore(X[:,i+2])

            return X, normalizers

        X = extract_features(lines, baseline, words)
        X, normalizers = normalize_training_data(X)
        Y = process_y_values(lines)
        return X,Y,normalizers


    def process_test_features(lines, baseline, words, normalizers):

        def normalize_test_data(X,normalizers):
            #print normalizers
            for i in range(X.shape[1]-2):
                if normalizers[i][1] == 0:
                    X[:,i+2] = [(x-normalizers[i][0]) for x in X[:,i+2]]
                else:
                    X[:,i+2] = [(x-normalizers[i][0])/(normalizers[i][1]) for x in X[:,i+2]]
                  
            return X



        X = extract_features(lines, baseline, words)
        X = normalize_test_data(X,normalizers)
        Y = process_y_values(lines)
        return X,Y


    #PROCESS INPUT FILE
    file_lines = read_file_lines(f_input)
    
    # READ HEADER
    header = file_lines[0]
    
    print "HEADER:" + str(header)
    
    # SPLIT TRAINING AND TEST
    lines_test = []
    lines_training = []
    for i in range(1,len(file_lines)):
        if i % 5 == 0:
            lines_test.append(file_lines[i])
        else:
            lines_training.append(file_lines[i])
                
    #lines_training = read_file_lines(f_training)
    #lines_test = read_file_lines(f_test)
    #print "TRAINING: " + str(len(lines_training)) 
    #print "TEST: " + str(len(lines_test))
    
    #For normalizing data
    baseline, words = select_features(lines_training, num_words)
    
    #print "BASELINE: " + str(baseline)
    
    print words
    feature_names = words
    X_train,Y_train,normalizers = process_training_features(lines_training, baseline, words)
    X_test,Y_test = process_test_features(lines_test, baseline, words, normalizers)

    return X_train, Y_train, X_test, Y_test, feature_names



#Cross-validation used for tuning model parameters
def fold(arr, K, i):
    N = len(arr)
    size = numpy.ceil(1.0 * N / K)
    arange = numpy.arange(N) # all indices
    heldout = numpy.logical_and(i * size <= arange, arange < (i+1) * size)
    rest = numpy.logical_not(heldout)
    return arr[heldout], arr[rest]


def kfold(arr, K):
    return [fold(arr, K, i) for i in range(K)]

################################ MAIN #######################

def main():
    
    #Process Data
    X_train, Y_train, X_test, Y_test, feature_names = read_data(f_input="data_gg_training.txt",num_words=50)

    #print X_test
    #print Y_train
    #print X_test
    #print Y_test

    #Train weight priors using cross-fold validation. Using 0.1 as default
    w_init_std = 0.1#train_w_init_std(X_train, Y_train, K=10)
    
    #Train weights
    w, f, d = train_w(X_train, Y_train,w_init_std)
    #w = w[0]
    print '--- Dict ---'
    print d
    print '--- Feature Weights ---'
    print w
    print 'INTERCEPT: ', w[0]
    for i in range(1,len(w)):
        print feature_names[i-1], '\t', w[i]

    print '---MSE and KL for Test Set ---'
    print 'MSE was', round(error(X_test, Y_test, w) * 10000,3) , 'x 1e4'
    print 'KL-divergence is', round(KL_divergence(X_test, Y_test, w) * 1000,3), 'x 1e3'
    print 'R2 score is', r2_score(Y_test,predict(X_test,w))

#     for i in range(len(Y_test)):
#         print Y_test[i], '\t', predict(X_test[i], w)

    
if __name__ == "__main__": main()
