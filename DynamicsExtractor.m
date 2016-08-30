% (c) Fangyu Wu
% July 2016
% Wrapper function for smoothing displacements and calculating velocities and accelerations.

function DynamicsExtractor
    extract_dynamics('test_A/displacements.mat')
    extract_dynamics('test_B/displacements.mat')
    %extract_dynamics('test_C/displacements.mat')
    extract_dynamics('test_D/displacements.mat')
end

% (c) Maria Laura Delle Monache
% July 2016 - Script to smooth displacements data from the vision algorithm

function [] = extract_dynamics(filename)
    disp(['Loading ', filename, '...'])
    load(filename)
    [N, T] = size(displacements(:,1:end,2)); % Number of vehicles and final time
    % Allocation arrays
    displacements_smoothed = zeros(N, T, 2);
    velocities_regularized = zeros(N, T, 2);
    accelerations_regularized = zeros(N, T, 2);
    for i =1:N
        veh_ID = num2str(i);
        disp('================================================================================')
        disp(['Smoohting displacements ', veh_ID, '...'])
        % Smooth horizontal displacements using RLOESS
        displacements_smoothed(i,:,1) = smooth(displacements(i,:,1), 0.1, 'rloess');
        % Smooth vertical displacements using RLOESS
        displacements_smoothed(i,:,2) = smooth(displacements(i,:,2), 0.1, 'rloess');
        disp('Differentiating for velocities...')
        % Compute horizontal velocity using the regularization function
        velocities_regularized(i,:,1) = Regularization_fcn(displacements_smoothed(i,:,1),0.01);
        % Compute vertical velocity using the regularization function
        velocities_regularized(i,:,2) = Regularization_fcn(displacements_smoothed(i,:,2),0.01);
        disp('Differentiating for accelerations...')
        % Compute horizontal acceleration using the regularization function
        accelerations_regularized(i,:,1) = Regularization_fcn(velocities_regularized(i,:,1),0.03);
        % Compute vertical acceleration using the regularization function
        accelerations_regularized(i,:,2) = Regularization_fcn(velocities_regularized(i,:,2),0.03);
        disp('================================================================================')
    end
    % Save results to file.
    disp('Saving to file...')
    save(strrep(filename, 'displacements', 'dynamics'), ... 
         'displacements_smoothed', ...
         'velocities_regularized', ...
         'accelerations_regularized')
end

% (C) Ke Han, 2015
% f represents the discrete values of the function to be differentiated,
% alpha is the weighting term in (4.39) of Piccoli et al. (2015). 

function [u] = Regularization_fcn(f ,alpha)
    f = f';
    dt=1/30; % time step
    n=length(f); 
    E=zeros(n);
    for i=1:n
        E(i+1:end,i)=1;
    end
    %% Differential operator
    D=zeros(n-1,n);
    for i=1:n-1
        D(i,i:i+1)=[-1,1];
    end
    D=D/dt;
    %% optimization
    H=2*alpha*(D')*D+2*(dt)^2*(E')*(E);
    c=2*dt*(f(1)-f)'*E; c=c';
    %option=optimset('Algorithm','active-set');
    [u,fval,flag]=quadprog(H,c);
    u = u';
end
