function run_matlab_tests()
%RUN_MATLAB_TESTS Exercise adapted methods with artificial inputs only.
project = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(project,'matlab'));
privateRoot = tempname;
mkdir(privateRoot);
cleanup = onCleanup(@() cleanupPrivate(privateRoot)); %#ok<NASGU>
cfg = config(privateRoot,4,struct('RUN_SLOW_ANALYSES',true, ...
    'svm',struct('BoxConstraint_grid',1,'KernelScale_grid_rbf',{{1}})));
failed = false;
try, config(project,4); catch, failed = true; end
assert(failed, 'Configuration allowed an output root inside the repository.');
[X,meta] = artificialFeatures();
import_feature_table(X,meta);
[P,O] = load_cached_data("both");
assert(height(P)==size(X,1) && height(O)*3==height(P));
assert(all(O.num_patches==3));
first = P.observation_key==O.observation_key(1);
assert(max(abs(O{1,cellstr("mean_band_"+string(1:4))}-mean(X(first,:),1))) < 1e-10);
[fold,~,K] = make_grouped_folds(P.subject_key,P.analysis_label,3,42);
for k=1:K
    assert(isempty(intersect(P.subject_key(fold==k),P.subject_key(fold~=k))));
end
bad = meta; bad.analysis_label(1)="class1";
failed=false; try, import_feature_table(X,bad); catch, failed=true; end
assert(failed,'Conflicting labels were accepted.');
evalc('quality = check_data_quality();');
assert(quality.patch_rows==height(P));
evalc('spectra = analyse_spectra();');
assert(~isempty(spectra));
evalc('embedding = analyse_pca_tsne();');
assert(embedding.valid_observations==height(O));
evalc('trajectory = analyse_tsne_time_trajectory();');
assert(trajectory.valid_domains==2);
evalc('classification = classify_by_time();');
assert(~isempty(classification.predictions));
evalc('transfer = validate_across_domains();');
assert(~isempty(transfer));
evalc('temporal = validate_across_time();');
assert(~isempty(temporal));
evalc('importance = analyse_band_importance();');
assert(~isempty(importance.permutation_importance));
U = importance.univariate_importance;
valid = isfinite(U.p_value);
assert(all(U.fdr_p_value(valid)>=U.p_value(valid)-1e-12 & U.fdr_p_value(valid)<=1));
% The prediction fixture is deliberately artificial and contains no fitted model outputs.
pred=P;
pred.predicted_label=pred.analysis_label;
pred.positive_score=2*double(pred.analysis_label=="class1")-1;
pred.fold=fold;
pred.model_type=repmat("synthetic_margin",height(P),1);
predictionFile=fullfile(privateRoot,'artificial_predictions.csv');
writetable(pred,predictionFile);
evalc('aggregation=evaluate_source_level(predictionFile);');
assert(height(aggregation.observation_predictions)==height(O));
% Test the newer model-comparison/transfer/top-k workflow with two model families.
options=struct('Models',["linear_svm","logistic"], 'NumFolds',2,'InnerFolds',2, ...
    'NumPermutations',1,'TopKValues',[1 4]);
evalc('study=run_model_study(options);');
assert(~isempty(study.WithinTopK) && ~isempty(study.Cross.Pairwise));
% Exercise the preserved v7.3/HDF5 inspection and extraction using generated records.
for label=["class0","class1"]
    select=O.analysis_label==label;
    obs=O(select,:); imageRecords=cell(height(obs),1);
    for i=1:height(obs)
        rows=P.observation_key==obs.observation_key(i);
        imageRecords{i}=struct('class',char(label),'domain',char(obs.domain(i)), ...
            'time',obs.time(i),'object_number',obs.object_number(i), ...
            'object_name',char(obs.subject_key(i)), 'patch_size',4, ...
            'total_patch_number',sum(rows),'channels',1:4, ...
            'locations_yx',zeros(sum(rows),2),'mean_values',P{rows,cellstr("band_"+string(1:4))});
    end
    save(fullfile(privateRoot,char(label+".mat")),'imageRecords','-v7.3');
end
evalc('inspection=inspect_mat_files();');
assert(inspection.class0.num_observations>0);
evalc('extraction=extract_feature_tables();');
[P2,O2]=load_cached_data("both");
assert(height(P2)==height(P) && height(O2)==height(O));
assert(isequal(sort(unique(P2.subject_key)),sort(unique(P.subject_key))));
evalc('pipeline=run_pipeline("quality");');
assert(all(pipeline.step_summary.status=="completed"));
fprintf('MATLAB workflow checks passed on artificial inputs.\n');
end

function [X,meta]=artificialFeatures()
rng(42);
X=[]; labels=strings(0,1); subjects=labels; observations=labels; domains=labels; times=[];
for d=1:2
    for t=1:2
        for s=(1:24)+(t-1)*16
            label="class"+string(mod(s-1,2));
            subject="source_"+string(d)+"_"+string(s);
            observation=subject+"_capture_"+string(t);
            X=[X;randn(3,4)+double(label=="class1")*[0.3 0.2 0 0]]; %#ok<AGROW>
            labels=[labels;repmat(label,3,1)]; %#ok<AGROW>
            subjects=[subjects;repmat(subject,3,1)]; %#ok<AGROW>
            observations=[observations;repmat(observation,3,1)]; %#ok<AGROW>
            domains=[domains;repmat("domain_"+string(d),3,1)]; %#ok<AGROW>
            times=[times;repmat(t,3,1)]; %#ok<AGROW>
        end
    end
end
meta=table(labels,subjects,observations,domains,times, ...
    'VariableNames',{'analysis_label','subject_key','observation_key','domain','time'});
end

function cleanupPrivate(folder)
close all;
clear config;
resolved=char(java.io.File(folder).getCanonicalPath());
temporary=char(java.io.File(tempdir).getCanonicalPath());
assert(startsWith(lower(resolved),[lower(temporary) filesep]),'Refusing cleanup outside the temporary directory.');
if isfolder(resolved), rmdir(resolved,'s'); end
end
