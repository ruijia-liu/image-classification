function cfg = config(privateRoot, channelCount, overrides)
%CONFIG Configure a private workspace and generic channel indices.
% Call config(privateRoot, channelCount) once in each MATLAB session.
% Optional overrides replace top-level fields, e.g. RUN_SLOW_ANALYSES.
persistent current
if nargin == 0
    if isempty(current)
        error('Call config(privateRoot, channelCount) before running an analysis.');
    end
    cfg = current;
    return;
end
validateattributes(channelCount, {'numeric'}, {'scalar','integer','>=',2});
root = char(java.io.File(char(privateRoot)).getCanonicalPath());
project = char(java.io.File(fileparts(fileparts(mfilename('fullpath')))).getCanonicalPath());
if strcmpi(root, project) || startsWith(lower(root), [lower(project) filesep])
    error('The private workspace must be outside this repository.');
end
if ~isfolder(root), error('Create the private workspace directory first.'); end
cfg = struct();
cfg.private_root = root;
cfg.input.files.class0 = struct('path', fullfile(root, 'class0.mat'), 'analysis_label', "class0");
cfg.input.files.class1 = struct('path', fullfile(root, 'class1.mat'), 'analysis_label', "class1");
cfg.paths.cache = fullfile(root, 'cache');
cfg.paths.figures = fullfile(root, 'outputs', 'figures');
cfg.paths.tables = fullfile(root, 'outputs', 'tables');
cfg.paths.predictions = fullfile(root, 'outputs', 'predictions');
cfg.paths.logs = fullfile(root, 'outputs', 'logs');
cfg.expected_channels = 1:channelCount;
cfg.RANDOM_SEED = 42;
cfg.FAST_MODE = true;
cfg.RUN_DOMAIN_TIME_TSNE = false;
cfg.RUN_SLOW_ANALYSES = false;
cfg.RUN_LONGITUDINAL_TIME_TRANSFER = false;
cfg.speed.outer_grouped_folds_fast = 3;
cfg.speed.inner_grouped_folds_fast = 2;
cfg.speed.max_patches_per_observation_fast = 30;
cfg.speed.auto_start_parallel_pool = false;
cfg.svm.kernels = ["linear", "rbf"];
cfg.svm.BoxConstraint_grid = [0.1, 1, 10];
cfg.svm.KernelScale_grid_rbf = {0.5, 1, 2, "auto"};
cfg.svm.standardize = true;
cfg.tsne.NumDimensions = 2;
cfg.tsne.Perplexity = 20;
cfg.tsne.InitialPCADimensions = min(10, channelCount);
cfg.analysis.score_threshold = 0; % SVM decision margin; set 0.5 for calibrated probabilities.
cfg.analysis.max_patches_per_observation = 30;
cfg.analysis.run_optional_slow_analyses = false;
cfg.analysis.outer_grouped_folds = 3;
cfg.analysis.inner_grouped_folds = 2;
if nargin >= 3
    names = fieldnames(overrides);
    for i = 1:numel(names)
        if any(strcmp(names{i}, {'paths','private_root','input','expected_channels'}))
            error('Private paths and channel schema cannot be overridden.');
        end
        cfg.(names{i}) = overrides.(names{i});
    end
end
current = cfg;
end
