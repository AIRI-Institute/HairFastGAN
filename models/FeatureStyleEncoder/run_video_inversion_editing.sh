VideoName='FP010363HD03'
Attribute='Heavy_Makeup'
Scale='1'
Sigma='3' # Choose appropriate gaussian filter size
VideoDir='./data/video/'
OutputDir='./output/video/'


# Cut video to frames
python video_processing.py --function 'video_to_frames' --video_path ${VideoDir}/${VideoName}.mp4 --output_path ${OutputDir} #--resize

# Crop and align the faces in each frame
python video_processing.py --function 'align_frames' --video_path ${VideoDir}/${VideoName}.mp4 --output_path ${OutputDir} --filter_size=${Sigma} --optical_flow

# Inversion
python test.py --config 143 --input_path ${OutputDir}/${VideoName}/${VideoName}_crop_align/ --save_path ${OutputDir}/${VideoName}/${VideoName}_inversion/

# Achieve latent manipulation
python video_processing.py --function 'latent_manipulation' --video_path ${VideoDir}/${VideoName}.mp4 --attr ${Attribute} --alpha=${Scale}

# Reproject the manipulated frames to the original video
python video_processing.py --function 'reproject_origin' --video_path ${VideoDir}/${VideoName}.mp4 --seamless
python video_processing.py --function 'reproject_manipulate' --video_path ${VideoDir}/${VideoName}.mp4 --attr ${Attribute} --seamless
python video_processing.py --function 'compare_frames' --video_path ${VideoDir}/${VideoName}.mp4 --attr ${Attribute} --strs 'Original,Projected,Manipulated'
