D = 'Depth\';
S = dir(fullfile(D,'P*'));
for ii = 1:numel(S)
    E = dir(fullfile(D,S(ii).name,'G*'));
    for jj = 1:numel(E)
        F = dir(fullfile(D,S(ii).name,E(jj).name,'*.txt'));
        for kk = 1:numel(F)
            N = fullfile(D,S(ii).name,E(jj).name,F(kk).name);
            img = load(N);
            save(fullfile('Depth_Preprocessed',S(ii).name, E(jj).name, strcat(extractBefore(F(kk).name,'.'), '.mat')), 'img')
        end
    end
end