

def makeGroupdict_custom(input_temp, output_temp):
    #input_temp_3

    import re
    #group size
    global_h = re.findall(r'\d+', output_temp.split('\n')[0])[1]
    global_w = re.findall(r'\d+', output_temp.split('\n')[0])[0]


    ##num_group
    group_num = []
    lines = input_temp.split('\n')
   # print(lines)
    for line in lines:
        if 'Number of people and objects of ' in line:
          #  print(line)
            try:
                group_num.append(line.split(':')[0].split('Group')[1])
            except:
                group_num.append('Non-group')
   # print('group_num',group_num)
    ##global_caption
    global_caption = input_temp.split('Global : ')[1].split(';')[0]

    ##group caption
    group_caption = []
    lines = input_temp.split('\n')
    for line in lines:
        if line.startswith('Group'):
            if 'bounding box' in line:
                continue
            
            if ' : 'in line:
                group_caption.append(line.split(' : ')[1].split(';'))


    ##group_bbox
    group_bbox = []
    lines = input_temp.split('\n')

    for line in lines:
        if ' : ' in line:
            continue
        if 'bounding box' not in line:
            continue
        if line.startswith('Group'):
            group_bbox_temp = line.split(';')[1]
            group_bbox_temp = re.findall(r'\d+', group_bbox_temp)
            group_bbox_temp = [int(num) for num in group_bbox_temp]
            group_bbox.append(group_bbox_temp)
            

    ##group number of person and obj
    lines = input_temp.split('\n')
    group_num_person = []
    group_num_obj = []
    for line in lines:
        if line.startswith('Number of people'):
            group_num_person.append(int(line.split(': P')[1].split(';')[0]))
            group_num_obj.append(int(line.split('; O')[1].split(';')[0]))
            
            
    ##group person

    lines = output_temp.split('\n')
    keypoint_temp_all = []
    keypoint_temp_group = []
    person_num = 0
    group_idx = 0
    for line in lines:
        if group_idx > len(group_num_person) -1:
            continue
        if group_num_person[group_idx] == 0:
            group_idx += 1
            keypoint_temp_all.append([])
        if line.startswith('P'):
            keypoint_temp = line.split(';')[0]
            keypoint_temp = re.findall(r'\d+', keypoint_temp)
            keypoint_temp = [int(num) for num in keypoint_temp]
            for k in range(18):
                keypoint_temp.insert(3*(k+1), 2)
            person_num+=1
            keypoint_temp_group.append(keypoint_temp[1:])
            if person_num == group_num_person[group_idx]:
                group_idx+=1
                keypoint_temp_all.append(keypoint_temp_group)
                keypoint_temp_group = []
                person_num = 0
                
    ##group person

    lines = output_temp.split('\n')
    person_bbox_temp_all = []
    person_bbox_temp_group = []
    person_num = 0
    group_idx = 0
    for line in lines:
        if group_idx > len(group_num_person) -1:
            continue
        if group_num_person[group_idx] == 0:
            group_idx += 1
            person_bbox_temp_all.append([])
        if line.startswith('P'):
            person_bbox_temp = line.split(';')[1]
            person_bbox_temp = re.findall(r'\d+', person_bbox_temp)
            person_bbox_temp = [int(num) for num in person_bbox_temp]
            # for k in range(18):
            #     keypoint_temp.insert(3*(k+1), 2)
            person_num+=1
            person_bbox_temp_group.append(person_bbox_temp)
            if person_num == group_num_person[group_idx]:
                group_idx+=1
                person_bbox_temp_all.append(person_bbox_temp_group)
                person_bbox_temp_group = []
                person_num = 0
                
            
    ##group obj

    lines = output_temp.split('\n')
    obj_temp_all = []
    obj_temp_group = []
    obj_num = 0
    group_idx = 0
    for line in lines:
        if group_idx > len(group_num_obj) -1:
            continue
        if group_num_obj[group_idx] == 0:
            group_idx += 1
            obj_temp_all.append([])
        if line.startswith('O'):
            obj_temp = line.split(';')[0]
            obj_temp = re.findall(r'\d+', obj_temp)
            obj_temp = [int(num) for num in obj_temp]
        #  for k in range(18):
        #      keypoint_temp.insert(3*(k+1), 2)
            obj_num+=1
            obj_temp_group.append(obj_temp[1:])
            if obj_num == group_num_obj[group_idx]:
                group_idx+=1
                obj_temp_all.append(obj_temp_group)
                obj_temp_group = []
                obj_num = 0        
            
    ## group person caption
    lines = input_temp.split('\n')
    person_caption_temp_all = []
    person_caption_temp_group = []
    person_num = 0
    group_idx = 0
    for line in lines:
        if group_idx > len(group_num_person) -1:
            continue
        if group_num_person[group_idx] == 0:
            group_idx += 1
            person_caption_temp_all.append([])
        if line.startswith('P'):
            person_caption_temp = line.split(': ')[1].split(';')[0]
            person_num+=1
            person_caption_temp_group.append(person_caption_temp)
            if person_num == group_num_person[group_idx]:
                group_idx+=1
                person_caption_temp_all.append(person_caption_temp_group)
                person_caption_temp_group = []
                person_num = 0   

    ## group obj caption
    lines = input_temp.split('\n')
    obj_caption_temp_all = []
    obj_caption_temp_group = []
    obj_num = 0
    group_idx = 0
    for line in lines:
        if group_idx > len(group_num_obj) -1:
            continue
        if group_num_obj[group_idx] == 0:
            group_idx += 1
            obj_caption_temp_all.append([])
        if line.startswith('O'):
            obj_caption_temp = line.split(': ')[1].split(';')[0]
            obj_num+=1
            obj_caption_temp_group.append(obj_caption_temp)
            if obj_num == group_num_obj[group_idx]:
                group_idx+=1
                obj_caption_temp_all.append(obj_caption_temp_group)
                obj_caption_temp_group = []
                obj_num = 0   
                
    group_dictionary = {}
    group_dictionary['shape'] = [int(global_w), int(global_h)]
    group_dictionary['global_caption'] = global_caption
    for idx, i in enumerate(group_num):
        if i == 'Non-group':
            i = 'non_group_person'
            group_dictionary[i] = {}
        else:
            group_dictionary[i] = {}
            group_dictionary[i]['group_bbox'] = {}
            group_bbox_temp = group_bbox[idx]
            group_dictionary[i]['group_bbox']['x'] = group_bbox_temp[0]
            group_dictionary[i]['group_bbox']['y'] = group_bbox_temp[1]
            group_dictionary[i]['group_bbox']['width'] = group_bbox_temp[2] - group_bbox_temp[0]
            group_dictionary[i]['group_bbox']['height'] = group_bbox_temp[3] - group_bbox_temp[1]
            
            group_dictionary[i]['group_caption'] = group_caption[idx]
        
        
        group_dictionary[i]['instance'] = {}
        for j in range(len(person_bbox_temp_all[idx])):
            group_dictionary[i]['instance'][str(j)] = {}
            group_dictionary[i]['instance'][str(j)] = {}
            group_dictionary[i]['instance'][str(j)]['x'] = person_bbox_temp_all[idx][j][0]
            group_dictionary[i]['instance'][str(j)]['y'] = person_bbox_temp_all[idx][j][1]
            group_dictionary[i]['instance'][str(j)]['width'] = person_bbox_temp_all[idx][j][2] - person_bbox_temp_all[idx][j][0]
            group_dictionary[i]['instance'][str(j)]['height'] = person_bbox_temp_all[idx][j][3] - person_bbox_temp_all[idx][j][1]
            group_dictionary[i]['instance'][str(j)]['caption'] = person_caption_temp_all[idx][j]
            group_dictionary[i]['instance'][str(j)]['keypoint'] = keypoint_temp_all[idx][j]
            
        group_dictionary[i]['obj'] = {}
        for j in range(len(obj_temp_all[idx])):
            group_dictionary[i]['obj'][str(j)] = {}
            group_dictionary[i]['obj'][str(j)] = {}
            group_dictionary[i]['obj'][str(j)]['x'] = obj_temp_all[idx][j][0]
            group_dictionary[i]['obj'][str(j)]['y'] = obj_temp_all[idx][j][1]
            group_dictionary[i]['obj'][str(j)]['width'] = obj_temp_all[idx][j][2] - obj_temp_all[idx][j][0]
            group_dictionary[i]['obj'][str(j)]['height'] = obj_temp_all[idx][j][3] - obj_temp_all[idx][j][1]
            group_dictionary[i]['obj'][str(j)]['center'] = \
            [obj_temp_all[idx][j][0] +  group_dictionary[i]['obj'][str(j)]['width']/ 2.0, obj_temp_all[idx][j][1]+ group_dictionary[i]['obj'][str(j)]['height']/ 2.0]
            group_dictionary[i]['obj'][str(j)]['caption'] = obj_caption_temp_all[idx][j]
    return group_dictionary
        