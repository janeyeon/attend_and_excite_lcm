import json
from _utils import *

def makeGroupdict(config):
    with open(config.json_path, "r") as st_json:
        keylayout_json = json.load(st_json)
    keylayout_data = keylayout_json[0]['images'][config.image_idx]

    ## global text

    ## group text and group instances

    # group instance number [['0', '1', '4', '5'], ['0', '1', '4', '5'], ['2', '3'], ['2', '3']]
    group_instances = keylayout_data['dense_caption']['sentences_region_idx']

    # group box
    group_box = keylayout_data['dense_caption']['union_boxes']

    group_dictionary = {}
    group_bbox_list = []
    for i in range(len(group_instances)):
        # print('i',i)
        if group_box[i] in group_bbox_list:
            ind = group_bbox_list.index(group_box[i])
            #  print(ind)
            group_dictionary[str(ind)]['group_caption'].append(
                keylayout_data['dense_caption']['sentences'][i]['raw'])
            group_bbox_list.append('pass')
            continue
        group_bbox_list.append(group_box[i])

        group_dictionary[str(i)] = {}
        group_dictionary[str(i)]['group_bbox'] = []

        group_dictionary[str(i)]['group_bbox'] = group_box[i]

        #     for j in group_box[i].keys():
        #         group_dictionary[str(i)]['group_bbox'].append(group_box[i][j]) # 'x', 'y', 'width', 'height'

        group_dictionary[str(i)]['instance'] = {}
        #  for set_list in group_instances:
        #     print(set_list)
        set_list = group_instances[i]
        for idxx, k in enumerate(set_list):
            try:
                bbox_temp = keylayout_data['box'][k]
            except:
                continue
            # print(k)
            group_dictionary[str(i)]['instance'][str(idxx)] = bbox_temp

        group_dictionary[str(i)]['group_instance_idx'] = set_list

        group_dictionary[str(i)]['group_caption'] = []
        group_dictionary[str(i)]['group_caption'].append(keylayout_data['dense_caption']['sentences'][i]['raw'])
        group_dictionary['obj'] = {}
        group_dictionary['obj'] = keylayout_data['obj_bbox']

    ## to remove the subset group
    group_instance_idx_list = []
    for i in group_dictionary.keys():
        if i == 'obj':
            continue
        group_instance_idx_list.append(group_dictionary[i]['group_instance_idx'])

    _, index_filtered = filter_contained_lists_with_indices(group_instance_idx_list)

    idx = 0

    removed_keys = []
    for i in group_dictionary.keys():
        if i == 'obj':
            continue
        if idx in index_filtered:
            removed_keys.append(i)
        idx += 1

    for i in removed_keys:
        del group_dictionary[i]

    ## center of obj boxes
    for i in group_dictionary['obj'].keys():
        x_center = group_dictionary['obj'][i]['x'] + group_dictionary['obj'][i]['width'] / 2.0
        y_center = group_dictionary['obj'][i]['y'] + group_dictionary['obj'][i]['height'] / 2.0

        group_dictionary['obj'][i]['center'] = [x_center, y_center]

    ## Put the obj into the group box by comparing the centor box of obj and group box

    for i in group_dictionary.keys():
        if i == 'obj':
            continue
        x = group_dictionary[i]['group_bbox']['x']
        y = group_dictionary[i]['group_bbox']['y']
        w = group_dictionary[i]['group_bbox']['width']
        h = group_dictionary[i]['group_bbox']['height']

        obj_num = 0
        group_dictionary[i]['obj'] = {}
        remove_list = []
        for j in group_dictionary['obj'].keys():  # [i]['center']
            center = group_dictionary['obj'][j]['center']
            center_x = center[0]
            center_y = center[1]
            if center_x > x and center_x < (x + w):
                if center_y > y and center_y < (y + h):
                    print('pass')
                    group_dictionary[i]['obj'][str(obj_num)] = group_dictionary['obj'][j]
                    obj_num += 1
                    remove_list.append(j)

    for i in remove_list:
        del group_dictionary['obj'][i]

    ## non-group people
    all_people_in_group = sorted(gather_elements_without_duplication(group_instances))
    all_people_in_image = list(keylayout_data['box'].keys())

    non_group = non_overlapping_elements(all_people_in_group, all_people_in_image)

    group_dictionary['non_group_person'] = {}
    group_dictionary['non_group_person']['instance'] = {}

    if len(non_group) != 0:
        non_person_num = 0
        for i in non_group:
            try:
                group_dictionary['non_group_person']['instance'][str(non_person_num)] = keylayout_data['box'][i]
                non_person_num += 1
            except:
                continue

    ## non-group object
    non_obj_num = 0
    group_dictionary['non_group_person']['obj'] = {}
    for i in group_dictionary['obj'].keys():
        group_dictionary['non_group_person']['obj'][str(non_obj_num)] = group_dictionary['obj'][i]
        non_obj_num += 1

    return group_dictionary, keylayout_data


def makeGroupdict_auto(config):
    with open(config.json_path, "r") as st_json:
        keylayout_json = json.load(st_json)
    keylayout_data = keylayout_json[0]['images'][config.image_idx]

    ## global text

    ## group text and group instances

    # group instance number [['0', '1', '4', '5'], ['0', '1', '4', '5'], ['2', '3'], ['2', '3']]
    group_instances = keylayout_data['dense_caption']['sentences_region_idx']

    # group box
    group_box = keylayout_data['dense_caption']['union_boxes']

    group_dictionary = {}
    group_bbox_list = []
    for i in range(len(group_instances)):
        # print('i',i)
        if group_box[i] in group_bbox_list:
            ind = group_bbox_list.index(group_box[i])
            #  print(ind)
            group_dictionary[str(ind)]['group_caption'].append(
                keylayout_data['dense_caption']['sentences'][i]['raw'])
            group_bbox_list.append('pass')
            continue
        group_bbox_list.append(group_box[i])

        group_dictionary[str(i)] = {}
        group_dictionary[str(i)]['group_bbox'] = []

        group_dictionary[str(i)]['group_bbox'] = group_box[i]

        #     for j in group_box[i].keys():
        #         group_dictionary[str(i)]['group_bbox'].append(group_box[i][j]) # 'x', 'y', 'width', 'height'

        group_dictionary[str(i)]['instance'] = {}
        #  for set_list in group_instances:
        #     print(set_list)
        set_list = group_instances[i]
        for idxx, k in enumerate(set_list):
            try:
                bbox_temp = keylayout_data['box'][k]
            except:
                continue
            # print(k)
            group_dictionary[str(i)]['instance'][str(idxx)] = bbox_temp

        group_dictionary[str(i)]['group_instance_idx'] = set_list

        group_dictionary[str(i)]['group_caption'] = []
        group_dictionary[str(i)]['group_caption'].append(keylayout_data['dense_caption']['sentences'][i]['raw'])
        group_dictionary['obj'] = {}
        group_dictionary['obj'] = keylayout_data['obj_bbox']

    ## to remove the subset group
    group_instance_idx_list = []
    for i in group_dictionary.keys():
        if i == 'obj':
            continue
        group_instance_idx_list.append(group_dictionary[i]['group_instance_idx'])

    _, index_filtered = filter_contained_lists_with_indices(group_instance_idx_list)

    idx = 0

    removed_keys = []
    for i in group_dictionary.keys():
        if i == 'obj':
            continue
        if idx in index_filtered:
            removed_keys.append(i)
        idx += 1

    for i in removed_keys:
        del group_dictionary[i]

    ## center of obj boxes
    for i in group_dictionary['obj'].keys():
        x_center = group_dictionary['obj'][i]['x'] + group_dictionary['obj'][i]['width'] / 2.0
        y_center = group_dictionary['obj'][i]['y'] + group_dictionary['obj'][i]['height'] / 2.0

        group_dictionary['obj'][i]['center'] = [x_center, y_center]

    ## Put the obj into the group box by comparing the centor box of obj and group box

    for i in group_dictionary.keys():
        if i == 'obj':
            continue
        x = group_dictionary[i]['group_bbox']['x']
        y = group_dictionary[i]['group_bbox']['y']
        w = group_dictionary[i]['group_bbox']['width']
        h = group_dictionary[i]['group_bbox']['height']

        obj_num = 0
        group_dictionary[i]['obj'] = {}
        remove_list = []
        for j in group_dictionary['obj'].keys():  # [i]['center']
            center = group_dictionary['obj'][j]['center']
            center_x = center[0]
            center_y = center[1]
            if center_x > x and center_x < (x + w):
                if center_y > y and center_y < (y + h):
                    print('pass')
                    group_dictionary[i]['obj'][str(obj_num)] = group_dictionary['obj'][j]
                    obj_num += 1
                    remove_list.append(j)

    for i in remove_list:
        del group_dictionary['obj'][i]

    ## non-group people
    all_people_in_group = sorted(gather_elements_without_duplication(group_instances))
    all_people_in_image = list(keylayout_data['box'].keys())

    non_group = non_overlapping_elements(all_people_in_group, all_people_in_image)

    group_dictionary['non_group_person'] = {}
    group_dictionary['non_group_person']['instance'] = {}

    if len(non_group) != 0:
        non_person_num = 0
        for i in non_group:
            try:
                group_dictionary['non_group_person']['instance'][str(non_person_num)] = keylayout_data['box'][i]
                non_person_num += 1
            except:
                continue

    ## non-group object
    non_obj_num = 0
    group_dictionary['non_group_person']['obj'] = {}
    for i in group_dictionary['obj'].keys():
        group_dictionary['non_group_person']['obj'][str(non_obj_num)] = group_dictionary['obj'][i]
        non_obj_num += 1

    return group_dictionary, keylayout_data


def getGlobalInfo(keylayout_data, config):
    global_text = keylayout_data['global_caption'][0]
    global_text = global_text.split('.')[0]+'.'
    global_prompt = global_text#.replace('people', '').replace('person', '').replace('man', '').replace('woman', '').replace('men', '').replace('women', '').replace('boy', '').replace('girl', '').replace('player', '')
    if not config.use_img2img :
        config.output_path = config.output_path / global_text[:100]
        config.output_path.mkdir(exist_ok=True, parents=True)

    global_H = int(keylayout_data['shape'][0]*config.zoom_ratio)
    global_W = int(keylayout_data['shape'][1]*config.zoom_ratio)
    

    global_canvas_H = int(max(64*math.ceil(global_H/64), 512))
    global_canvas_W = int(max(64*math.ceil(global_W/64), 512))

    print(f'(global_H,global_W): {(global_H,global_W)}')
    print(f'(global_canvas_H,global_canvas_W): {(global_canvas_H,global_canvas_W)}')

    return global_prompt, global_H, global_W, global_canvas_H, global_canvas_W


def getGroupIds(group_dictionary, config):
    group_ids = list(group_dictionary.keys())
    group_ids.remove('obj')
    if not config.use_nongroup:
        group_ids.remove('non_group_person')
    return group_ids

def getGroupIds_auto(group_dictionary):
    group_ids = list(group_dictionary.keys())
    group_ids.remove('shape')
    group_ids.remove('global_caption')
    return group_ids

def get_canvas(H, W):
    global_canvas_H = int(max(64*math.ceil(H/64), 512))
    global_canvas_W = int(max(64*math.ceil(W/64), 512))

    return global_canvas_H, global_canvas_W