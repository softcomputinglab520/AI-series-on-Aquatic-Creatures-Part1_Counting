import numpy as np
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis

def createtarget(bnd, bc, classname, start_frame):
    return {
        'bnd': [bnd],
        'bc': [bc],
        'classname': [classname],
        'state': 'consider',
        'istracking': True,
        'pre_time': 0,
        'start': start_frame,
        'behavior':'normal',
        'checkpoint':'no',
        'behavior_num':'0',
        'num':'0'
    }

def initialtargets(targets, bodys, start_frame):
    for i in range(len(bodys)):
        targets[str(i)] = createtarget(bodys[i][1], np.array(
            [(bodys[i][1][0] + bodys[i][1][2]) / 2, (bodys[i][1][1] + bodys[i][1][3]) / 2]), bodys[i][0], start_frame)

def caloverlap1(input, ref):
    if input[0] > ref[0]:
        xmin0, ymin0, xmax0, ymax0= ref
        xmin1, ymin1, xmax1, ymax1 = input
    else:
        xmin0, ymin0, xmax0, ymax0 = input
        xmin1, ymin1, xmax1, ymax1 = ref
    if xmin0 <= xmax1 and xmax1 <= xmax0:
        x = xmax0 - xmin1
    elif xmin1 <= xmin0 and xmax0 <= xmax1:
        x = xmax0 - xmin0
    elif xmin0 <= xmin1 and xmax1 <= xmax0:
        x = xmax1 - xmin1
    else:
        x = xmax1 - xmin0

    if ymin1 <= ymax0 and ymax0 <= ymax1:
        y = ymax0 - ymin1
    elif ymin1 <= ymin0 and ymax0 <= ymax1:
        y = ymax0 - ymin0
    elif ymin0 <= ymin1 and ymax1 <= ymax0:
        y = ymax1 - ymin1
    else:
        y = ymax1 - ymin0
    # temp = (x * y) / (((xmax0 - xmin0) * (ymax0 - ymin0)) + ((xmax1 - xmin1) * (ymax1 - ymin1)) - (x * y))
    return (x * y) / (((xmax0 - xmin0) * (ymax0 - ymin0)) + ((xmax1 - xmin1) * (ymax1 - ymin1)) - (x * y))


def caloverlap01(input, ref):
    if input[0] < ref[0]:
        xmin0, ymin0, xmax0, ymax0 = ref
        xmin1, ymin1, xmax1, ymax1 = input
    else:
        xmin0, ymin0, xmax0, ymax0 = input
        xmin1, ymin1, xmax1, ymax1 = ref
    if xmin0 <= xmax1 and xmax1 <= xmax0:
        x = xmax0 - xmin1
    elif xmin1 <= xmin0 and xmax0 <= xmax1:
        x = xmax0 - xmin0
    elif xmin0 <= xmin1 and xmax1 <= xmax0:
        x = xmax1 - xmin1
    else:
        x = xmax1 - xmin0

    if ymin1 <= ymax0 and ymax0 <= ymax1:
        y = ymax0 - ymin1
    elif ymin1 <= ymin0 and ymax0 <= ymax1:
        y = ymax0 - ymin0
    elif ymin0 <= ymin1 and ymax1 <= ymax0:
        y = ymax1 - ymin1
    else:
        y = ymax1 - ymin0
    # temp = (x * y) / (((xmax0 - xmin0) * (ymax0 - ymin0)) + ((xmax1 - xmin1) * (ymax1 - ymin1)) - (x * y))
    return (x * y) / (((xmax0 - xmin0) * (ymax0 - ymin0)) + ((xmax1 - xmin1) * (ymax1 - ymin1)) - (x * y))

def checktrackstate(targets, num):
    # lens = [len(targets[str(x)]['bc']) for x in targets]
    # max_len = max(lens)
    for t in range(len(targets)):
        tl = len(targets[t]['bc'])
        if tl + targets[t]['start'] < num or 5 < targets[t]['pre_time']:
            targets[t]['istracking'] = False

def addtarget(targets, candidate):
        if len(candidate) != 0:
            targets.append(candidate)

def targetpredict(targets, num):
    # lens = [len(targets[str(x)]['bc']) for x in targets]
    # max_len = max(lens)
    for tt in range(len(targets)):
        # temp5.append(len(targets[str(tt)]['bc']))
        if targets[tt]['istracking'] :
            t_len = len(targets[tt]['bc'])
            if num == t_len + targets[tt]['start']:
                if len(targets[tt]['bc']) < 5:
                    bc = targets[tt]['bc'][-1]
                    bnd = targets[tt]['bnd'][-1]
                    targets[tt]['pre_time'] += 1
                else:
                    bc0 = targets[tt]['bc'][t_len - 2]
                    bc1 = targets[tt]['bc'][t_len - 1]
                    bnd0 = targets[tt]['bnd'][t_len - 2]
                    bnd1 = targets[tt]['bnd'][t_len - 1]
                    
#                    print('bc0:',bc0,'bc1:',bc1)
                    speedx = bc1[0] - bc0[0]
                    speedy = bc1[1] - bc0[1]
                    # bc = bc1 + 0 * speed1
                    speedx = 0.5 * speedx
                    speedy = 0.5 * speedy
                    xmin = int(bnd1[0] + speedx)
                    ymin = int(bnd1[1] + speedy)
                    xmax = int(bnd1[2] + speedx)
                    ymax = int(bnd1[3] + speedy)
                    bnd = [xmin, ymin, xmax, ymax]
                    bc = [(xmin + xmax) / 2, (ymin + ymax) / 2]
                    targets[tt]['pre_time'] += 1
                classname = targets[tt]['classname'][-1]
                targets[tt]['bnd'].append(bnd)
                targets[tt]['bc'].append(bc)
                targets[tt]['classname'].append(classname)
            else:
                targets[tt]['pre_time'] = 0




def matchtarget03(targets, bodys, num,Abnormal_behavior):
    temp = []
    temp4 = []
    for b1 in bodys:
#        print('matchtarget03-00')
        classname = b1[-1]
        temp1 = []
        ol_temp = []
        bc1 = np.array([(b1[0] + b1[2]) / 2, (b1[1] + b1[3]) / 2])
        bnd1 = b1[:4]
        for b in range(len(targets)):
#            print('matchtarget03-01')
            if targets[b]['istracking']:
                bc = np.array(targets[b]['bc'][-1])
                bnd = targets[b]['bnd']
                temp1.append((np.sqrt(np.inner(bc - bc1, bc - bc1).tolist()).tolist()))
                ol_temp.append(caloverlap1(bnd1, bnd[0]))
            else:
                temp1.append(2000)
                ol_temp.append(0)
        temp.append(temp1)
        bc = np.array(targets[b]['bc'][-1])
        distx = int(abs((bc[0]-bc1[0])))
        disty = int(abs((bc1[1]-bc[1])))
#        print(distx,disty)
        if Abnormal_behavior == 1 and num >= (len(targets[b]['classname'])+targets[b]['start']):
            if len(temp1) != 0 and min(temp1) < 100 and len(ol_temp) != 0 and min(ol_temp) > 0.5 and distx < 20 and disty < 20:
                targets[b]['bnd'].append([int(bnd1[0]),int(bnd1[1]),int(bnd1[2]),int(bnd1[3])])
                bcint = bc1.tolist()
                bcint = [int(bcint[0]),int(bcint[1])]
                targets[b]['bc'].append(bcint)
                if classname == 0:
                    targets[b]['classname'].append( targets[b]['classname'][-1])
                else:
                    targets[b]['classname'].append(classname)
#                print('A=10')
            elif max(distx,disty) < 30 and num >= (len(targets[b]['classname'])+targets[b]['start']):
                targets[b]['bnd'].append([int(bnd1[0]),int(bnd1[1]),int(bnd1[2]),int(bnd1[3])])
                bcint = bc1.tolist()
                bcint = [int(bcint[0]),int(bcint[1])]
                targets[b]['bc'].append(bcint)
                if classname == 0:
                    targets[b]['classname'].append( targets[b]['classname'][-1])
                else:
                    targets[b]['classname'].append(classname)#                print('A=11')
            else:
                temp4 = createtarget(bnd1, bc1.tolist(), b1[-1], num)
#                print('A=12')
        else:
             if len(temp1) != 0 and min(temp1) < 100 and len(ol_temp) != 0 and min(ol_temp) > 0.3 and distx < 20 and disty < 20 and num >= (len(targets[b]['classname'])+targets[b]['start']):
                targets[b]['bnd'].append([int(bnd1[0]),int(bnd1[1]),int(bnd1[2]),int(bnd1[3])])
                bcint = bc1.tolist()
                bcint = [int(bcint[0]),int(bcint[1])]
                targets[b]['bc'].append(bcint)
                if classname == 0:
                    targets[b]['classname'].append( targets[b]['classname'][-1])
                else:
                    targets[b]['classname'].append(classname)#                print('A=00')
             elif max(distx,disty) < 30 and num >= (len(targets[b]['classname'])+targets[b]['start']):
                targets[b]['bnd'].append([int(bnd1[0]),int(bnd1[1]),int(bnd1[2]),int(bnd1[3])])
                bcint = bc1.tolist()
                bcint = [int(bcint[0]),int(bcint[1])]
                targets[b]['bc'].append(bcint)
                if classname == 0:
                    targets[b]['classname'].append( targets[b]['classname'][-1])
                else:
                    targets[b]['classname'].append(classname)#                print('A=01')
    return temp4

def matching_beh(log_list,frame_count):
    just_x = []
    ans = ['normal']
    log = '0'
    error_code = 0
    beh_model=[[ [10,20,21,11,81,80,10],'weird swimming'],
                [[20,21,11,81,20],'dying'],
                [[51,41,40,50,60,61,51],'weird swimming'],
                [[41,40,50,60,41],'dying'],
#                [[10,11],'exhausted'],
                [[51,50],'exhausted'],
#                [[10,11,10],'weird swimming'],
                [[51,50,51],'weird swimming']]
    
    print(log_list)

    for i in range(len(log_list[0])):
#        print(log_list[0][i])
        if log_list[0][i] == 10:
            log_list[0][i] = 4232
        if log_list[0][i] == 20:
            log_list[0][i] = 8464
        if log_list[0][i] == 30:
            log_list[0][i] = 5708
        if log_list[0][i] == 40:
            log_list[0][i] = 11416
        if log_list[0][i] == 50:
            log_list[0][i] = 3962
        if log_list[0][i] == 60:
            log_list[0][i] = 7924
        if log_list[0][i] == 70:
            log_list[0][i] = 4118
        if log_list[0][i] == 80:
            log_list[0][i] = 8236
        if log_list[0][i] == 11:
            log_list[0][i] = 4352
        if log_list[0][i] == 21:
            log_list[0][i] = 8704
        if log_list[0][i] == 31:
            log_list[0][i] = 5678
        if log_list[0][i] == 41:
            log_list[0][i] = 11356
        if log_list[0][i] == 51:
            log_list[0][i] = 3842
        if log_list[0][i] == 61:
            log_list[0][i] = 7684
        if log_list[0][i] == 71:
            log_list[0][i] = 4148
        if log_list[0][i] == 81:
            log_list[0][i] = 8296
#        print(log_list)
        
    for i in range(len(beh_model)):
        for j in range(len(beh_model[i][0])):
            if beh_model[i][0][j] == 10:
                beh_model[i][0][j] = 4232
            if beh_model[i][0][j] == 20:
                beh_model[i][0][j] = 8464
            if beh_model[i][0][j] == 30:
                beh_model[i][0][j] = 5708
            if beh_model[i][0][j] == 40:
                beh_model[i][0][j] = 11416
            if beh_model[i][0][j] == 50:
                beh_model[i][0][j] = 3962
            if beh_model[i][0][j] == 60:
                beh_model[i][0][j] = 7924
            if beh_model[i][0][j] == 70:
                beh_model[i][0][j] = 4118
            if beh_model[i][0][j] == 80:
                beh_model[i][0][j] = 8236
            if beh_model[i][0][j] == 11:
                beh_model[i][0][j] = 4352
            if beh_model[i][0][j] == 21:
                beh_model[i][0][j] = 8704
            if beh_model[i][0][j] == 31:
                beh_model[i][0][j] = 5678
            if beh_model[i][0][j] == 41:
                beh_model[i][0][j] = 11356
            if beh_model[i][0][j] == 51:
                beh_model[i][0][j] = 3842
            if beh_model[i][0][j] == 61:
                beh_model[i][0][j] = 7684
            if beh_model[i][0][j] == 71:
                beh_model[i][0][j] = 4148
            if beh_model[i][0][j] == 81:
                beh_model[i][0][j] = 8296
#        print(beh_model)
    for i in range(len(log_list)):
#        print('matching_beh00')
        for j in range(len(beh_model)):
            if len(log_list[i]) <= 15:
                error_code = 1
                break
            
#            print('matching_beh01')
#            print(beh_model[j][0])
#            if i > 0:
#                if log_list[i-1]==11:
#                    if log_list[i] == 81:
#                        log_list[i] = 1
#                if log_list[i-1] == 81:
#                    if log_list[i] == 11:
#                       log_list[i] = 91
#                if log_list[i-1] == 91:
#                    log_list[i-1] = 11
#                if log_list[i-1] == 1:
#                    log_list[i-1] = 81

            dis = int(dtw.distance(beh_model[j][0],log_list[i]))
            path = dtw.warping_path(beh_model[j][0], log_list[i])
            dtwvis.plot_warping(beh_model[j][0], log_list[i], path, dis, filename=("warp"+str(frame_count)+str(i)+str(j)+".png"))
            just_x.append([dis,j,beh_model[j][1]])
        if error_code == 1:
            error_code = 0
            break
        temp = [k[0] for k in just_x]
        min_index = np.argmin(temp)
        print('2:',just_x[min_index][0])
#        print(just_x[min_index][0]<30)
        if just_x[min_index][0] < 300:
            ans=([just_x[min_index][1]])
            log = just_x[min_index][2]
        elif 300 < just_x[min_index][0] < 1000:
            ans=['Suspected abnormal']
            log = '0'        
        else:
            ans=['normal']
            log = '0'
        just_x = []
#        ans = []
    print('3:',ans,log)
    return ans,log

dis = int(dtw.distance([10,11,10],[40,41,0,40,0]))


