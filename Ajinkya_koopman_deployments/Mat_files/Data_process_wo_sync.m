myDir = 'D:\pysindy\Teleop_data\raw_teleop_data'; %gets directory
saveDir = 'D:\pysindy\Teleop_data\filtered_teleop_data';
myFiles = dir(fullfile(myDir,'*.mat'));
for k = 1:length(myFiles)
    baseFileName = myFiles(k).name;
    fullFileName = fullfile(myDir, baseFileName);
    fprintf(1, 'Now reading %s\n', fullFileName);
    data = load(fullFileName);
    states_tsec = second(datetime(data.states_ts.Time,'ConvertFrom','posixtime', 'Format', 'yyyy-MM-dd HH:mm:SSSS'));
    states_tsec = time_cont(states_tsec); states_tsec = states_tsec-states_tsec(1);
    inputs_tsec = second(datetime(data.inputs_ts.Time,'ConvertFrom','posixtime', 'Format', 'yyyy-MM-dd HH:mm:SSSS'));
    inputs_tsec = time_cont(inputs_tsec); inputs_tsec = inputs_tsec-inputs_tsec(1);
    [lim_, index_] = min(abs(states_tsec-inputs_tsec(end)));
    states_filtered = data.states(1:index_,:);
    states_ts_filtered = states_tsec(1:index_);

   updated_data.inputs = data.inputs;
   updated_data.inputs_ts = inputs_tsec;
   updated_data.states = states_filtered;
   updated_data.states_ts = states_ts_filtered;

   base_name_ = split(string(baseFileName),'.'); %Extract base file name
   fname_updated = string(base_name_(1) + '_filtered.mat'); % Save file as
   save_file_dir = string(saveDir+fname_updated);
   fprintf(1,'Saving updated file as %s\n',save_file_dir)
   save(save_file_dir,'updated_data')

end


%% Function
function input_data = time_cont(input_data)
    for i=2:length(input_data)
        if input_data(i) < input_data(i-1)
            input_data(i) = input_data(i) + 60;
        end
    end
end