import numpy as np
import bpy
from mathutils import Vector
import pickle
import bmesh
import glob
import os
import time
import sys


def save_state(ms,meshes,vtx_picks,co_picks,moves,convert=False,as_init=False):
    base_dir = bpy.context.scene.base_dir_path+'/'
    if convert:
        for mesh in meshes: print('vertices:',len(mesh.vertices))
        np_meshes = np.array([[[v.co[0],v.co[1],v.co[2]] for v in mesh.vertices] for mesh in meshes])
    else:
        np_meshes = meshes
    state = {'ms':ms,
             'meshes': np_meshes,
             'vtx_picks': vtx_picks,
             'co_picks': co_picks,
             'moves': moves}
    if as_init:
        with open(base_dir+'init.pickle', mode='wb') as f:
            pickle.dump(state,f)
    with open(base_dir+'state[pid'+str(os.getpid())+'].pickle', mode='wb') as f:
        pickle.dump(state,f)


def load_state(cloth):
    base_dir = bpy.context.scene.base_dir_path+'/'
    try:
        state=pickle.load(open(base_dir+'state[pid'+str(os.getpid())+'].pickle', mode='rb'))
        ms = state['ms']
        meshes = state['meshes']
        vtx_picks = state['vtx_picks']
        co_picks = state['co_picks']
        moves = state['moves']
        return ms,meshes,vtx_picks,co_picks,moves
    except IOError:
        print('LOAD FAILURE')
        return 0,np.array([]),[],[],[]


def load_blank_state(cloth):
    base_dir = bpy.context.scene.base_dir_path+'/'
    try:
        state=pickle.load(open(base_dir+'init.pickle', mode='rb'))
        meshes = state['meshes']
        return meshes
    except IOError:
        print('*'*60)
        print('ERROR: failed to load initial state:')
        print(base_dir+'init.pickle')
        print('ensure that the base path field is set correctly and that init.pickle exists there.')
        print('*'*60)
        return np.array([])


def assign(cloth,np_mesh):
    print('assign np_mesh with shape', np_mesh.shape, 'to cloth.data.vertices of size', len(cloth.data.vertices))
    for i in range(len(cloth.data.vertices)):
        for c in range(3):
            cloth.data.vertices[i].co[c] = np_mesh[i,c]


def convert_to_np(mesh,flatten=False):
    if flatten:
        n_vtx = int(len(mesh.vertices)/2)
        np_mesh = np.empty((n_vtx,3))
        for i in range(n_vtx):
            for c in range(3):
                np_mesh[i,c] = (mesh.vertices[i].co[c]+mesh.vertices[i+n_vtx].co[c])/2
    else:
        n_vtx = int(len(mesh.vertices))
        np_mesh = np.empty((n_vtx,3))
        for i in range(n_vtx):
            np_mesh[i] = np.array(mesh.vertices[i].co)
    return np_mesh


# find graspable edges given graspable edge width
def get_edge_vtx_list(vtxs,edge_width):
    
    # rasterise
    res = 64
    edge_width = edge_width*(.7*res/34) # 5 cm
    edge_width_int = int(np.ceil(edge_width))
    
    n_vtxs = vtxs.shape[0]
    vtxs = vtxs.copy() * bpy.context.scene.scaler
    
    if bpy.context.scene.periodic:
        vtxs[:,0] = (vtxs[:,0]+1)%2-1
        vtxs[:,1] = (vtxs[:,1]+1)%2-1
    
    vtxs[:,0] = (vtxs[:,0]+1)/2*res
    vtxs[:,1] = (vtxs[:,1]+1)/2*res
    
    vtxs = vtxs.round().astype(np.int32)
    vtxs = np.clip(vtxs,0,res-1)
    flat = np.zeros([res,res],dtype=np.float32)
    flat[vtxs[:,0],vtxs[:,1]] = 1
    
    # make convolution kernel for edge measurement
    cx,cy = np.mgrid[-edge_width_int:edge_width_int+1,-edge_width_int:edge_width_int+1]
    d = (cx**2+cy**2)**.5<edge_width
    print(np.where(d,1,0))
    fd = np.reshape(d,-1)
    fx,fy = np.reshape(cx,-1),np.reshape(cy,-1)
    fx,fy = fx[fd],fy[fd]
    
    outside = 1-flat
    edges = np.zeros(flat.shape)
    
    # convolve (would be nicer with scipy but good luck importing scipy in a blender context...)
    for ix in range(res):
        for iy in range(res):
            if flat[ix,iy]:
                edges[ix,iy] = outside[(ix+fx)%res,(iy+fy)%res].sum()>0
    
    edge_vtxs = []
    for i in range(n_vtxs):
        if edges[int(vtxs[i,0]),int(vtxs[i,1])]:
            edge_vtxs.append(i)
            
    print('edge vtxs:',len(edge_vtxs),'/',n_vtxs)
    return edge_vtxs


# build the Manipulation tab in the Blender GUI
class PANEL_PT_ClothManipulationPanel(bpy.types.Panel):
    
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_context = "objectmode"
    bl_category = "Manipulation"
    bl_label = "Cloth Manipulation"

    def draw(self, context):
        
        col = self.layout.column(align=True)
        
        col.prop(context.scene, "base_dir_path")
        col.prop(context.scene, "data_dir")
        col.prop(context.scene, "run_name")
        row = col.row()
        row.prop(context.scene, "periodic")
        row.prop(context.scene, "rounded_trajectory_prop")
        row.prop(context.scene, "polar")
        row = col.row()
        row.prop(context.scene, "edge_mode")
        row.prop(context.scene, "edge_width")
        row.prop(context.scene, "pick_top_layer_only")
        row = col.row()
        row.label(text='Motion speed')
        row.prop(context.scene, "speed")
        row = col.row()
        row.label(text='Pick-up range')
        row.prop(context.scene, "pickup_range_prop")
        row = col.row()
        row.label(text='Stabilisation frames')
        row.prop(context.scene, "stabilisation_frames")
        row = col.row()
        row.label(text='Maximum displacement')
        row.prop(context.scene, "max_movement")
        
        box = col.box()
        col2 = box.column()
        col2.label(text='Direct controls')
        col2.operator("mesh.init_cloth_manipulation", text="Initialise cloth manipulation")
        col2.operator("mesh.step_cloth_manipulation", text="Run one random manipulation")
        col2.operator("mesh.step_cloth_manipulation_given", text="Run specified manipulation")
        row = col2.row()
        col3 = row.column()
        col3.prop(context.scene, "move_left_exists")
        col3.prop(context.scene, "move_left_x")
        col3.prop(context.scene, "move_left_y")
        col3 = row.column()
        col3.prop(context.scene, "move_right_exists")
        col3.prop(context.scene, "move_right_x")
        col3.prop(context.scene, "move_right_y")
        col3 = row.column()
        col3.label(text="displacement")
        col3.prop(context.scene, "move_slide_x")
        col3.prop(context.scene, "move_slide_y")
        row = col2.row()
        row.operator("mesh.rotate_cloth", text="Rotate cloth")
        row.prop(context.scene, "user_specified_rotation")
        #split = col2.split(align=False,percentage=0.9)
        split = col2.split(factor=0.9,align=False)
        split.label(text='Save current state as initial state:')
        split.operator("mesh.save_as_init", text="!!")
        
        box = col.box()
        col2 = box.column()
        col2.label(text='Data generation')
        col2.operator("mesh.full_cloth_manipulation", text="Generate data")
        row = col2.row()
        col2.prop(context.scene, "length_prop")
        row = col2.row()
        row.prop(context.scene, "files_prop")
        row.prop(context.scene, "batch_prop")
        col2.prop(context.scene, "name_base")
        
        box = col.box()
        col2 = box.column()
        col2.label(text='Play back stored sequences')
        row = col2.row()
        row.operator("mesh.manipulation_playback", text="Run playback")
        row.prop(context.scene, "inferred_prop")
        col2.prop(context.scene, "data_subset")
        row = col2.row()
        row.label(text='Data range:')
        row.prop(context.scene, "playback_range_data_from")
        row.prop(context.scene, "playback_range_data_to")
        
        row = col2.row()
        row.prop(context.scene, "playback_single")
        col3 = row.column()
        col3.prop(context.scene, "playback_data")
        col3.prop(context.scene, "playback_seq_len")
        col3.prop(context.scene, "playback_from_step")
        col3.prop(context.scene, "playback_step")
        
        box = col.box()
        col2 = box.column()
        col2.label(text='Ping-Pong mode')
        col2.operator("mesh.pingpong", text="START")
        row = col2.row()
        row.prop(context.scene, "pingpong_timeout")
        row.prop(context.scene, "pingpong_halt")


class SaveAsInit(bpy.types.Operator):
    
    bl_idname = "mesh.save_as_init"
    bl_label = "Save current state as initial state"
    
    def invoke(self, context, event):
        print('saving current state as initial state...')
        cloth = bpy.data.objects['cloth']
        save_state(0,convert_to_np(cloth.data,flatten=False)[None,...],[],[],[],as_init=True)
        print('ok!')
        return {'FINISHED'}


class ClothManipulationInit(bpy.types.Operator):
    
    bl_idname = "mesh.init_cloth_manipulation"
    bl_label = "Initialise cloth manipulation"
    bl_options = {"UNDO"}
    
    def invoke(self, context, event):
        
        print('init')
        sequence_length = context.scene.length_prop
        frames_per_manipulation = 100
        cloth = bpy.data.objects['cloth']
        meshes = load_blank_state(cloth)

        cloth.animation_data_clear()
        cloth.vertex_groups.clear()

        try:
            cloth.modifiers.remove(cloth.modifiers['mod'])
        except KeyError:
            pass

        assign(cloth,meshes[0])

        meshes = np.array([[[v.co[0],v.co[1],v.co[2]] for v in cloth.data.vertices]])

        hand = bpy.data.objects['hand']
        hand.animation_data_clear()
        hand.location = (0,0,context.scene.base_z)
        hand.rotation_euler = (0,0,0)
        bpy.context.scene.frame_set(0)

        pick_seq = []
        move_seq = []
        state_seq = []

        frame_num = 0
        frame_num_key = 0
        simulation_time_steps = frames_per_manipulation*sequence_length

        ms = 0
        failures = 0
        data_mods = []
        
        if not context.scene.edge_mode:
            vtx_picks = np.random.choice(len(cloth.data.vertices),(sequence_length,2))
            for s in range(sequence_length):
                blank = np.random.randint(4)
                if blank < 2: vtx_picks[s,blank] = -1
            print('vtx picks:')
            print(vtx_picks)
        else: vtx_picks = []
        co_picks = np.random.uniform(-1.0,1.0,(sequence_length,2,2))
        if context.scene.polar:
            move_dists = np.random.uniform(0.0,context.scene.max_movement,[sequence_length])
            move_angles = np.random.uniform(0,360,[sequence_length])
            print('move distances:',move_dists)
            print('move angles:',move_angles)
            moves = np.stack([move_dists*np.cos(move_angles),move_dists*np.sin(move_angles)],1)
        else:
            moves = np.random.uniform(-context.scene.max_movement,context.scene.max_movement,[sequence_length,2])
        print('saving moves:')
        print(moves)
        
        save_state(ms,meshes,vtx_picks,co_picks,moves)
        return {'FINISHED'}


class RotateCloth(bpy.types.Operator):

    bl_idname = "mesh.rotate_cloth"
    bl_label = "Rotate cloth by given amount"
    bl_options = {"UNDO"}

    def invoke(self, context, event):
        cloth = bpy.data.objects['cloth']
        ms,meshes,vtx_picks,co_picks,moves = load_state(cloth)
        mesh = convert_to_np(cloth.to_mesh())#scene = bpy.context.scene, apply_modifiers = True, settings = 'RENDER'))
        angles = np.arctan2(mesh[:,1],mesh[:,0])
        hypots = np.hypot(mesh[:,1],mesh[:,0])
        angles += np.deg2rad(context.scene.user_specified_rotation)
        mesh[:,0] = hypots * np.cos(angles) # new x coords
        mesh[:,1] = hypots * np.sin(angles) # new y coords
        assign(cloth,mesh)
        print(meshes.shape,mesh.shape)
        meshes[-1] = mesh
        save_state(ms,meshes,vtx_picks,co_picks,moves)
        return {'FINISHED'}


class ClothManipulationStepGiven(bpy.types.Operator):
    
    bl_idname = "mesh.step_cloth_manipulation_given"
    bl_label = "Run specified manipulation"
    bl_options = {"UNDO"}
    
    def invoke(self, context, event):
        bpy.context.scene.move_given = True
        ClothManipulationStep.invoke(self,context,event)
        bpy.context.scene.move_given = False
        return {'FINISHED'}


class ClothManipulationStep(bpy.types.Operator):
    
    bl_idname = "mesh.step_cloth_manipulation"
    bl_label = "Run one manipulation step"
    bl_options = {"UNDO"}
    
    def invoke(self, context, event):
        
        baseZ = context.scene.base_z
        wait_time = 0

        cloth = bpy.data.objects['cloth']
        ms, meshes, vtx_picks, co_picks, fixed_moves = load_state(cloth)
        if ms == len(fixed_moves):
            print('*'*60)
            print('To run more than', ms, 'manipulations, increase sequence length and reinitialise.')
            print('*'*60)
            return {'FINISHED'}
        move = fixed_moves[ms]
                        
        # clear animation
        cloth.animation_data_clear()
        cloth.vertex_groups.clear()

        hand = bpy.data.objects['hand']
        hand.animation_data_clear()

        assign(cloth,meshes[ms])
        
        pick_list = []
        vtxs = meshes[ms]
        
        if context.scene.move_given: # if move given by user
            
            if context.scene.move_left_exists: # check whether left arm picks
                co_picks[ms,0] = np.array([context.scene.move_left_x,context.scene.move_left_y])
            else:
                co_picks[ms,0] = np.array([np.nan,np.nan])
            
            if context.scene.move_right_exists: # check whether right arm picks
                co_picks[ms,1] = np.array([context.scene.move_right_x,context.scene.move_right_y])
            else:
                co_picks[ms,1] = np.array([np.nan,np.nan])
            move = np.array([context.scene.move_slide_x,context.scene.move_slide_y])
            
            if context.scene.edge_mode:
                print('adjusting move for edge mode')
                adjusted_move = None
                edge_vtxs = get_edge_vtx_list(vtxs,context.scene.edge_width)
                for p in range(2):
                    if np.isnan(co_picks[ms,p]).any():continue
                    min_dp = 99999
                    for v in edge_vtxs:
                        dp = co_picks[ms,p]-vtxs[v,:2]
                        dp = dp[0]**2+dp[1]**2
                        mx,my = move[0],move[1]
                        x0,y0 = vtxs[v,0],vtxs[v,1]
                        x1,y1 = co_picks[ms,p,0],co_picks[ms,p,1]
                        x2,y2 = co_picks[ms,p,0]+mx,co_picks[ms,p,1]+my
                        dm = abs(my*x0-mx*y0+x2*y1-y2*x1)/((my**2+mx**2)**.5)
                        if dm<0.05 and dp+dm<min_dp:
                            nearest_v = v
                            min_dp = dp
                    c = vtxs[nearest_v]
                    co_drop = co_picks[ms,p]+move
                    co_picks[ms,p] = c[:2]
                    if adjusted_move is None:
                        adjusted_move = co_drop-co_picks[ms,p]
                    else:
                        adjusted_move = (adjusted_move+(co_drop-co_picks[ms,p]))/2
                print('unadjusted move:', move)
                move = np.array(adjusted_move)
                print('adjusted move:', move)
            
        elif not context.scene.edge_mode and vtx_picks != []:
            
            for p in range(2):
                if vtx_picks[ms][p] >= 0:
                    c = vtxs[vtx_picks[ms][p]]
                    co_picks[ms,p] = c[:2]
                    print('vertex', p, 'found at', co_picks[ms][p])
                else:
                    co_picks[ms,p] = [np.nan,np.nan]
            
        elif context.scene.edge_mode:
            
            edge_vtxs = get_edge_vtx_list(vtxs,context.scene.edge_width)
            blank = np.random.randint(4)
            for p in range(2):
                if blank == p:
                    co_picks[ms,p] = [np.nan,np.nan]
                else:
                    v = np.random.randint(len(edge_vtxs))
                    co_picks[ms][p] = vtxs[edge_vtxs[v],:2]

        print('picking by coordinates:')
        print(co_picks[ms])
        print('slide by:')
        print(move)

        periodic_unit = 2/context.scene.scaler
        if context.scene.pick_top_layer_only:
            for p in range(2):
                pick = -1
                max_z = -9999
                min_d = 9999
                for i in range(vtxs.shape[0]):
                    if not np.isnan(co_picks[ms][p]).any():
                        if context.scene.periodic:
                            dx = abs(vtxs[i,0]-co_picks[ms][p][0])%periodic_unit
                            dy = abs(vtxs[i,1]-co_picks[ms][p][1])%periodic_unit
                            d = (dx**2+dy**2)**.5
                        else:
                            d = ((vtxs[i,0]-co_picks[ms][p][0])**2+(vtxs[i,1]-co_picks[ms][p][1])**2)**.5
                        z = vtxs[i,2]
                        if d < context.scene.pickup_range_prop and z >= max_z:
                            if z > max_z or d < min_d:
                                pick = i
                                max_z = z
                                min_d = d
                if pick >= 0:
                    pick_list.append(pick)
        else:
            for i in range(vtxs.shape[0]):
                for p in range(2):
                    if not np.isnan(co_picks[ms][p]).any():
                        if context.scene.periodic:
                            dx = abs(vtxs[i,0]-co_picks[ms][p][0])%periodic_unit
                            dy = abs(vtxs[i,1]-co_picks[ms][p][1])%periodic_unit
                            d = (dx**2+dy**2)**.5
                        else:
                            d = ((vtxs[i,0]-co_picks[ms][p][0])**2+(vtxs[i,1]-co_picks[ms][p][1])**2)**.5
                        if d < context.scene.pickup_range_prop: pick_list.append(i)
            
        print('pick list:', pick_list)
        
        if pick_list == []:
            result_mesh = vtxs
        else:
            vg = cloth.vertex_groups.new(name='vg')
            vg_add = cloth.vertex_groups.new(name='vg_add')

            # weight paint the pick-up vertices
            vg_add.add(pick_list,1.0,'REPLACE')
            
            simulation_type = ['cloth','softbody'][0]
            if simulation_type == 'cloth': cloth.modifiers["Cloth"].settings.vertex_group_mass = 'vg'
            else: cloth.modifiers["Softbody"].settings.vertex_group_goal = 'vg_add'
            cloth.modifiers["VertexWeightMix"].vertex_group_a = 'vg'
            cloth.modifiers["VertexWeightMix"].vertex_group_b = 'vg_add'
            cloth.modifiers["VertexWeightMix"].mix_mode = 'ADD' # should be set already
            cloth.modifiers["VertexWeightMix"].mix_set = 'OR' # should be set already
            
            frame_num = 0
            hand.location = (0,0,baseZ)
            hand.keyframe_insert(data_path="location",frame=frame_num)
            
            cloth.modifiers["VertexWeightMix"].mask_constant = 0
            cloth.keyframe_insert(data_path='modifiers["VertexWeightMix"].mask_constant',frame=frame_num)
            
            frame_num += 1
            
            cloth.modifiers["VertexWeightMix"].mask_constant = 1
            cloth.keyframe_insert(data_path='modifiers["VertexWeightMix"].mask_constant',frame=frame_num)
            
            # wait time
            frame_num += 1
            
            # computing movement trajectory
            move_length = (move[0]**2+move[1]**2)**.5
            if context.scene.rounded_trajectory_prop:
                drop_height = .25#0.28
                tan_a = drop_height/move_length
                dist_start_to_drop = (drop_height**2+move_length**2)**.5
                move_time = int(np.ceil(dist_start_to_drop/(context.scene.speed*0.05)))
                arc_diagonal = (dist_start_to_drop**2+(tan_a*dist_start_to_drop)**2)**.5
                unit_move = (move[0]/move_length,move[1]/move_length)
                pivot = (arc_diagonal*unit_move[0]/2,arc_diagonal*unit_move[1]/2)
                radius = (pivot[0]**2+pivot[1]**2)**.5
                arc = 2*np.arcsin(0.5*dist_start_to_drop/radius)
                for i in range(0,move_time+1):
                    f = float(i/move_time)
                    z = baseZ+np.sin(f*arc)*radius
                    p = -np.cos(f*arc)
                    x = pivot[0]+p*move[0]/2
                    y = pivot[1]+p*move[1]/2
                    hand.location = (x,y,z)
                    hand.keyframe_insert(data_path="location",frame=frame_num)
                    frame_num += 1
            else:
                x = 0
                y = 0
                z = baseZ
                
                xy_step = 0.02*context.scene.speed
                z_step = 0.02*context.scene.speed
                
                lift_height = 0.15
                move_normal = move/move_length
                
                for i in range(0,100):
                    print('move',move)
                    x = np.sign(move[0])*min(abs(x+move_normal[0]*xy_step),abs(move[0]))
                    y = np.sign(move[1])*min(abs(y+move_normal[1]*xy_step),abs(move[1]))
                    z = min(z+z_step,baseZ+lift_height)
                    print('hand to',x,y,z,'move:',move)
                    hand.location = (x,y,z)
                    hand.keyframe_insert(data_path="location",frame=frame_num)
                    frame_num += 1
                    if x == move[0] and y == move[1]: break
                    
            cloth.modifiers["VertexWeightMix"].mask_constant = 1
            cloth.keyframe_insert(data_path='modifiers["VertexWeightMix"].mask_constant',frame=frame_num)
            
            frame_num += 1
            release_at = frame_num
            
            cloth.modifiers["VertexWeightMix"].mask_constant = 0
            cloth.keyframe_insert(data_path='modifiers["VertexWeightMix"].mask_constant',frame=frame_num)

            frame_num += 1
            hand.location = (x,y,baseZ)
            hand.keyframe_insert(data_path="location",frame=frame_num)

            frame_num += context.scene.stabilisation_frames

            n_vtxs = vtxs.shape[0]
            
            t_start = time.time()
            for i in range(frame_num+1):
                t_frame = time.time()
                bpy.context.scene.frame_set(i)
                print('play',i,'/',frame_num, 'time:', time.time()-t_frame)
                #bpy.context.scene.update()
                
                if i == release_at:
                    print('release')
            
            bpy.data.scenes["Scene"].frame_end = frame_num
            
            print('manipulation time:', time.time()-t_start)
                
            # make snapshot of resulting mesh
            result_mesh = convert_to_np(cloth.to_mesh())
        
        if meshes.shape[0] <= ms+1:
            print('concatenating result mesh')
            print('vtx ranges', result_mesh.min(0), result_mesh.max(0))
            print(meshes.shape, result_mesh.shape)
            meshes = np.concatenate([meshes,[result_mesh]])
            
        ms += 1        

        save_state(ms,meshes,vtx_picks,co_picks,fixed_moves,convert=False)
        print('ok!')
        return {'FINISHED'}


class ClothManipulationSequence(bpy.types.Operator):
    
    bl_idname = "mesh.full_cloth_manipulation"
    bl_label = "Run and save full manipulation sequence"
    
    def invoke(self, context, event):
        
        base_dir = context.scene.base_dir_path+'/'
        
        file_index = 0
        dir_str = base_dir+context.scene.data_dir+'/'
        os.makedirs(dir_str, exist_ok=True)
        file_list = glob.glob(dir_str+context.scene.name_base+'??????.npz')
        if len(file_list):
            last_file_name = max(file_list)
            file_index = int(last_file_name[-10:-4])+1
        else: file_index = 0
        print('')
        print('starting at file index: ',file_index)
        
        for f in range(context.scene.files_prop):
            meshes_list = []
            picks_list = []
            moves_list = []
            cloth = bpy.data.objects['cloth']
            
            for b in range(context.scene.batch_prop):
                ClothManipulationInit.invoke(self,context,event)
                print('  making sequence', b, end=' ', flush=True)
                for m in range(context.scene.length_prop):
                    ClothManipulationStep.invoke(self,context,event)
                    print('.', end='', flush=True)
                print('', flush=True) # newline
                ms,meshes,vtx_picks,co_picks,moves = load_state(cloth)
                print('loaded co_picks:', co_picks)
                meshes_list.append(np.array(meshes).copy())
                picks_list.append(np.array(co_picks).copy())
                moves_list.append(np.array(moves).copy())
                
            dir_str = base_dir+context.scene.data_dir+'/'
            np.savez(dir_str+context.scene.name_base+str(file_index).zfill(6),
                    states=np.array(meshes_list),
                    picks=np.array(picks_list),
                    moves=np.array(moves_list))
            print('manipulation sequence batch',f,'saved')
            file_index += 1
            
        print('finished')
        return {'FINISHED'}
 
 
class ClothManipulationPlayback(bpy.types.Operator):
    
    bl_idname = "mesh.manipulation_playback"
    bl_label = "Load an inferred manipulation sequence"
    
    def invoke(self, context, event):
        
        cloth = bpy.data.objects['cloth']
        
        found = False
        base_dir = context.scene.base_dir_path+'/'
        run_name = context.scene.run_name
        cloud_file_dir = context.scene.data_dir
        
        data_set = context.scene.data_subset
        output_path = base_dir+'/'+run_name+'/eval/execution_results'
        os.makedirs(output_path,exist_ok=True)
        output_path = output_path+'/'+data_set
        os.makedirs(output_path,exist_ok=True)
        
        # get planning data
        if context.scene.inferred_prop:
            plan_file_name = base_dir+'/'+run_name+'/eval/plans_'+data_set+'.pickle'
            print('load file:',plan_file_name)
            try:
                f=open(plan_file_name,'rb')
                plan_data = pickle.load(f,encoding='latin1')
                plan_moves = plan_data['moves']
                plan_angles = plan_data['angles']
                plan_mirror = plan_data['mirror']
            except FileNotFoundError:
                print('File Not Found:',plan_file_name)
                return {'FINISHED'}
        
        # get data set
        print('loading data set:', data_set)
        state_list = []
        move_list = []
        pick_list = []
        file_counter = 0
        print('pattern:', base_dir+'/'+cloud_file_dir+'/'+data_set+'/???????.npz')
        file_list = glob.glob(base_dir+'/'+cloud_file_dir+'/'+data_set+'/???????.npz')
        n_files = len(file_list)
        print('first file',file_list[0])
        for file_name in file_list:
            try: f = open(file_name,mode='rb')
            except IOError: break
            data = np.load(f)
            state_list.append(data['states'])
            pick_list.append(data['picks'])
            move_list.append(data['moves'])
        state_data = np.concatenate(state_list,0)
        pick_data = np.concatenate(pick_list,0)
        move_data = np.concatenate(move_list,0)
        print('states shape:', state_data.shape)
        print('picks shape:', pick_data.shape)
        print('moves shape:', move_data.shape)
        n_data = state_data.shape[0]
        nr_length = len(str(n_data))
        
        # execute plans
        max_sequence_length = 3
        
        if context.scene.playback_single:
            range_data = [context.scene.playback_data]
            range_seq_len = [context.scene.playback_seq_len]
            range_from_step = [context.scene.playback_from_step]
            range_step = range(context.scene.playback_step)
        else:
            range_data = range(context.scene.playback_range_data_from,context.scene.playback_range_data_to+1)
        
        for data_index in range_data:
            
            if not data_index in range_data: continue
            
            print(' data index:', data_index)
            counter = 0
            result_dict = {}
            moves_dict = {}
            
            for seq_len in range(1,max_sequence_length+1):
                print('  sequence length:', seq_len)
                
                skipping = 0
                if context.scene.playback_single and not seq_len in range_seq_len:
                    print('skipping sequence length',seq_len)
                    skipping = 1
                
                for from_step in range(max_sequence_length-seq_len+1):
                    
                    if skipping:
                        counter += 1
                        continue
                    
                    if context.scene.playback_single and not from_step in range_from_step:
                        print('skipping starting step',from_step)
                        counter += 1
                        continue
                    
                    # look up mesh
                    mesh = state_data[data_index,from_step]
                    
                    # rotate mesh
                    if context.scene.inferred_prop:
                        try:
                            rot = plan_angles[(data_index,seq_len,from_step)]
                        except KeyError:
                            print('key error:', (data_index,seq_len,from_step))
                            print('valid keys:',plan.angles.keys)
                            
                        angles = np.arctan2(mesh[:,1],mesh[:,0])
                        hypots = np.hypot(mesh[:,1],mesh[:,0])
                        angles += rot
                        mesh[:,0] = hypots * np.cos(angles) # new x coords
                        mesh[:,1] = hypots * np.sin(angles) # new y coords
                    
                    # split moves data into picks and slide
                    if context.scene.inferred_prop:
                        moves = plan_moves[(data_index,seq_len,from_step)]
                        moves /= context.scene.scaler
                        i = 0
                        moves_dict[(data_index,seq_len,from_step)] = moves
                        picks = np.array([[[moves[s,i+1],moves[s,i+0]],[moves[s,i+3],moves[s,i+2]]] for s in range(seq_len)])
                        slide = np.array([[moves[s,i+5],moves[s,i+4]] for s in range(seq_len)])
                        print('moves:')
                        print(moves)
                        print('picks:')
                        print(picks)
                        print('slide:')
                        print(slide)
                    else:
                        picks = pick_data[data_index,from_step:from_step+seq_len]
                        slide = move_data[data_index,from_step:from_step+seq_len]
                        moves_dict[(data_index,seq_len,from_step)] = np.concatenate((picks[:,0],picks[:,1],slide),1)
                    
                    save_state(0,np.array([mesh]),[],picks,slide,False)
                    
                    for step in range(seq_len):
                        if context.scene.playback_single and not step in range_step:
                            print('skipping step',step)
                            continue
                        ClothManipulationStep.invoke(self,context,event)
                    ms,meshes,_,_picks,_moves = load_state(cloth)
                    meshes_array = np.array(meshes)
                    print('meshes_array:',meshes_array.shape)
                    result_dict[(data_index,seq_len,from_step)] = meshes_array
                    counter += 1
                    
            # save outcomes
            print('results:')
            for key in result_dict.keys():
                print(key,':',result_dict[key].shape)
                
            save_dict = {'results': result_dict, 'moves': moves_dict}
            
            # marker to distinguish task replay from plan execution
            marker = '' if context.scene.inferred_prop else 't'
            with open(output_path+'/'+marker+str(data_index).zfill(5)+'.pickle','wb') as outfile:
                pickle.dump(save_dict, outfile, protocol=2)
                
        print('playback finished and outcomes saved')
        return {'FINISHED'}


class PingPong(bpy.types.Operator):
    
    bl_idname = "mesh.pingpong"
    bl_label = "Perform first step of a manipulation sequence - for ping pong evaluation"
    
    def invoke(self, context, event):
        
        cloth = bpy.data.objects['cloth']
        
        base_dir = context.scene.base_dir_path+'/'
        run_name = context.scene.run_name
        pp_path = base_dir+run_name+'/eval/pingpong/'
        print('path:', pp_path)
        
        wait_time = context.scene.pingpong_timeout
        while 1:
        
            file_list = glob.glob(pp_path+'/*p.pickle')
            if not len(file_list):
                sys.stdout.write('  waiting... '+str(wait_time)+' \r')
                sys.stdout.flush()
                wait_time -= 1
                if wait_time == 0:
                    print('timeout         ')
                    break # exit script after (over) a minute of inactivity
                time.sleep(1)
                continue
            
            plan_file_name = file_list[0]
            print('load file:',plan_file_name)
            try:
                pp_state = pickle.load(open(plan_file_name,'rb'),encoding='latin1')
                os.remove(plan_file_name)
            except FileNotFoundError:
                print('File Not Found:',plan_file_name)
                return {'FINISHED'}
            except EOFError:
                print('EOF error on file:',plan_file_name,'--> removing broken file')
                os.remove(plan_file_name)
                continue
            
            mesh = pp_state['mesh']
            
            # split moves data into picks and slide
            moves = pp_state['move']
            moves /= context.scene.scaler
            
            i = 0
            picks = np.array([[[moves[1],moves[0]],[moves[3],moves[2]]]])
            slide = np.array([[moves[5],moves[4]]])

            print('moves:')
            print(moves)
            print('picks:')
            print(picks)
            print('slide:')
            print(slide)
            
            save_state(0,np.array([mesh]),[],picks,slide,False)
        
            ClothManipulationStep.invoke(self,context,event)
            ms,meshes,_,_picks,_moves = load_state(cloth)
            pp_state = {'mesh':meshes[-1]}
            
            # save ping pong state
            plan_file_name = plan_file_name[:-8]+'b.pickle'
            print('save to:',plan_file_name)
            with open(plan_file_name,'wb') as outfile:
                pickle.dump(pp_state,outfile,protocol=2)
                
            if context.scene.pingpong_halt: break
            wait_time = context.scene.pingpong_timeout
                
        return {'FINISHED'}

    
def register():
    
    # register methods
    bpy.utils.register_class(SaveAsInit)
    bpy.utils.register_class(ClothManipulationInit)
    bpy.utils.register_class(ClothManipulationStep)
    bpy.utils.register_class(ClothManipulationStepGiven)
    bpy.utils.register_class(ClothManipulationSequence)
    bpy.utils.register_class(ClothManipulationPlayback)
    bpy.utils.register_class(PingPong)
    bpy.utils.register_class(PANEL_PT_ClothManipulationPanel)
    bpy.utils.register_class(RotateCloth)
    
    bpy.types.Scene.length_prop = bpy.props.IntProperty(name = "Sequence length",
        description = "Manipulation sequence length",default = 3)
    bpy.types.Scene.batch_prop = bpy.props.IntProperty(name = "Examples per file",
        description = "Number of sequences per file",default = 1)
    bpy.types.Scene.files_prop = bpy.props.IntProperty(name = "Number of files",
        description = "Number of files to make",default = 1000)
    bpy.types.Scene.rounded_trajectory_prop = bpy.props.BoolProperty(name = "Rounded",
        description = "On: rounded trajectory. Off: straight trajectory",default = True)
    bpy.types.Scene.pick_top_layer_only = bpy.props.BoolProperty(name = "Pick top",
        description = "On: pick top cloth layer only. Off: pick all cloth layers.",default = False)
    bpy.types.Scene.polar = bpy.props.BoolProperty(name = "Polar",
        description = "On: generate displacements as (length,angle). Off: generate displacements as (dx,dy).",default = True)
    bpy.types.Scene.periodic = bpy.props.BoolProperty(name = "Periodic",
        description = "Periodic boundary conditions on/off",default = False)
    bpy.types.Scene.edge_mode = bpy.props.BoolProperty(name = "Edge mode",
        description = "Whether to pick anywhere or just at the edges (of the cloth's silhouette)",default = False)
    bpy.types.Scene.edge_width = bpy.props.FloatProperty(name = "Edge width",
        description = "Width of edge used in edge mode (in centimeters, measures from cloth's silhouette edge)",default = 5.0)
    bpy.types.Scene.max_movement = bpy.props.FloatProperty(name = "",
        description = "Maximum movement distance",default = 1.4)
    bpy.types.Scene.speed = bpy.props.FloatProperty(name = "",
        description = "Speed of the manipulation motion",default = 1.0)
    bpy.types.Scene.pickup_range_prop = bpy.props.FloatProperty(name = "",
        description = "Vertices within this range from pickup point are held by gripper",default = 0.025)
    
    # dir paths
    bpy.types.Scene.base_dir_path = bpy.props.StringProperty(name = "Base path",
        description = "Base directory for saving and loading files",default = '<base_path>')
    bpy.types.Scene.data_dir = bpy.props.StringProperty(name = "Data dir",
        description = "Dir to load mesh and move data from",default = '<data_dir>')
    
    # save file name base
    bpy.types.Scene.name_base = bpy.props.StringProperty(name = "Name base",
        description = "File name base for generated examples to",default = '<name_base>')
    
    # for loading and playback
    bpy.types.Scene.run_name = bpy.props.StringProperty(name = "Run name",
        description = "Run to load manipulation from",default = '<run_name>')
    enum_items = (('train','Train',''),('test','Test',''))
    bpy.types.Scene.data_subset = bpy.props.EnumProperty(name = "Subset", items = enum_items,
        description = "Data subset to use")
    
    # playback range settings
    bpy.types.Scene.inferred_prop = bpy.props.BoolProperty(name = "Inferred",
        description = "On: load inferred manipulation. Off: load original manipulation.",default = True)
    bpy.types.Scene.playback_single = bpy.props.BoolProperty(name = "Run single sequence:",
        description = "On: run sequence specified below. Off: run all sequences in range..",default = False)
    bpy.types.Scene.playback_range_data_from = bpy.props.IntProperty(name = "",
        description = "Playback data range: from index",default = 0)
    bpy.types.Scene.playback_range_data_to = bpy.props.IntProperty(name = "",
        description = "Playback data range: to index (inclusive)",default = 0)
    bpy.types.Scene.playback_data = bpy.props.IntProperty(name = "data index",
        description = "Playback data index",default = 0)
    bpy.types.Scene.playback_seq_len = bpy.props.IntProperty(name = "sequence length",
        description = "Playback sequence length",default = 1)
    bpy.types.Scene.playback_from_step = bpy.props.IntProperty(name = "starting step",
        description = "Playback start step",default = 0)
    bpy.types.Scene.playback_step = bpy.props.IntProperty(name = "step",
        description = "Playback steps",default = 1)
    
    # misc simulation settings
    bpy.types.Scene.scaler = bpy.props.FloatProperty(name = "scaler",
        description = "cloth scaler",default = 1.0)
    bpy.types.Scene.base_z = bpy.props.FloatProperty(name = "base Z",
        description = "base z coordinate",default = 0.05)
    bpy.types.Scene.stabilisation_frames = bpy.props.IntProperty(name = "",
        description = "Number of frame to resolve after releasing grasp",default = 10)
    
    # settings for user-provided move
    bpy.types.Scene.move_given = bpy.props.BoolProperty(name = "use user-defined move:",
        description = "use user-provided  move",default = False)
    bpy.types.Scene.move_right_exists = bpy.props.BoolProperty(name = "right active",
        description = "right hand pick on/off",default = True)
    bpy.types.Scene.move_right_x = bpy.props.FloatProperty(name = "x",
        description = "right hand x coordinate",default = 0.0)
    bpy.types.Scene.move_right_y = bpy.props.FloatProperty(name = "y",
        description = "right hand y coordinate",default = 0.0)
    bpy.types.Scene.move_left_exists = bpy.props.BoolProperty(name = "left active",
        description = "left hand pick on/off",default = True)
    bpy.types.Scene.move_left_x = bpy.props.FloatProperty(name = "x",
        description = "left hand x coordinate",default = 0.0)
    bpy.types.Scene.move_left_y = bpy.props.FloatProperty(name = "y",
        description = "left hand y coordinate",default = 0.0)
    bpy.types.Scene.move_slide_x = bpy.props.FloatProperty(name = "x",
        description = "move distance x",default = 0.0)
    bpy.types.Scene.move_slide_y = bpy.props.FloatProperty(name = "y",
        description = "move distance y",default = 0.0)
    bpy.types.Scene.user_specified_rotation = bpy.props.FloatProperty(name = "degrees",
        description = "degrees to rotate by",default = 45.0)
    
    # ping pong settings
    bpy.types.Scene.pingpong_halt = bpy.props.BoolProperty(name = "halt",
        description = "On: Halt after processing a single step. Off: Resume waiting after step completion.",default = False)
    bpy.types.Scene.pingpong_timeout = bpy.props.IntProperty(name = "timeout",
        description = "Maximum time to wait for a task",default = 120)

if __name__ == "__main__" :
    register()
