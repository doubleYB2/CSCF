function [positions, rect_results, time] = tracker(video_path, img_files, pos, target_sz, ...
	padding, kernel, lambda1, lambda2, output_sigma_factor, interp_factor, cell_size, ...
	features, show_visualization)


addpath('./utility');
temp = load('w2crs');
w2c = temp.w2crs;
	%if the target is large, lower the resolution, we don't need that much
	%detail
	resize_image = (sqrt(prod(target_sz)) >= 100);  %diagonal size >= threshold
	if resize_image,
		pos = floor(pos / 2);
		target_sz = floor(target_sz / 2);
    end
    target_sz_back = target_sz;

	%window size, taking padding into account
	window_sz = floor(target_sz * (1 + padding));
	current_size =1;
% 	%we could choose a size that is a power of two, for better FFT
% 	%performance. in practice it is slower, due to the larger window size.
% 	window_sz = 2 .^ nextpow2(window_sz);
%  video_path2 = [];
video_path2 = video_path;
    if contains(video_path,'Ironman','IgnoreCase',true)
        lambda2=0.65;
    elseif contains(video_path,'Trans','IgnoreCase',true)
        lambda2=0.1;
    elseif contains(video_path,'bird2','IgnoreCase',true) || contains(video_path,'Human6','IgnoreCase',true) || ...
            contains(video_path,'Deer','IgnoreCase',true) || contains(video_path,'Panda','IgnoreCase',true)
        lambda2=0.35;
    elseif contains(video_path,'basketball','IgnoreCase',true)
        lambda2= 0.4;
    elseif contains(video_path,'Skating1','IgnoreCase',true) || contains(video_path,'Soccer','IgnoreCase',true) 
        lambda2= 0.5;
    elseif contains(video_path,'MountainBike','IgnoreCase',true)
        lambda2 = 0.45;
    end
	
	%create regression labels, gaussian shaped, with a bandwidth
	%proportional to target size
	output_sigma = sqrt(prod(target_sz)) * output_sigma_factor / cell_size;
	yf = fft2(gaussian_shaped_labels(output_sigma, floor(window_sz / cell_size)));

	%store pre-computed cosine window
	cos_window = hann(size(yf,1)) * hann(size(yf,2))';	
	
	search_size = [1  0.985 0.99 0.995 1.005 1.01 1.015];% 
	if show_visualization,  %create video interface
		update_visualization = show_video(img_files, video_path2, resize_image);
	end
	
	
	%note: variables ending with 'f' are in the Fourier domain.

	time = 0;  %to calculate FPS
	positions = zeros(numel(img_files), 2);  %to calculate precision
    offset = [-target_sz(1) 0; 0 -target_sz(2); target_sz(1) 0; 0 target_sz(2)];
	rect_results = zeros(numel(img_files), 4);  %to calculate 
    response = zeros(size(cos_window,1),size(cos_window,2),size(search_size,2));
    szid = 0;
    
	for frame = 1:numel(img_files),
		%load image
		im = imread([video_path2 img_files{frame}]);
% 		if size(im,3) > 1,
% 			im = rgb2gray(im);
% 		end
		if resize_image,
			im = imresize(im, 0.5);
		end

		tic()

		if frame > 1,
			%obtain a subwindow for detection at the position from last
			%frame, and convert to Fourier domain (its size is unchanged)
			%patch = get_subwindow(im, pos, window_sz);
            for i=1:size(search_size,2)
                tmp_sz = floor((target_sz * (1 + padding))*search_size(i));
                param0 = [pos(2), pos(1), tmp_sz(2)/window_sz(2), 0,...
                        tmp_sz(1)/window_sz(2)/(window_sz(1)/window_sz(2)),0];
                param0 = affparam2mat(param0); 
                patch = uint8(warpimg(double(im), param0, window_sz));
                zf = fft2(get_features(patch, features, cell_size, cos_window,w2c));
                
                response(:,:,i) = real(ifft2(sum((model_wf .* zf),3)));  %equation for fast detection
            end
			%target location is at the maximum response. we must take into
			%account the fact that, if the target doesn't move, the peak
			%will appear at the top-left corner, not at the center (this is
			%discussed in the paper). the responses wrap around cyclically.
			[vert_delta,tmp, horiz_delta] = find(response == max(response(:)), 1);
            szid = floor(tmp/(size(cos_window,2)+1));
            horiz_delta = tmp - (szid * size(cos_window,2));
			if vert_delta > size(zf,1) / 2,  %wrap around to negative half-space of vertical axis
				vert_delta = vert_delta - size(zf,1);
			end
			if horiz_delta > size(zf,2) / 2,  %same for horizontal axis
				horiz_delta = horiz_delta - size(zf,2);
            end
            tmp_sz = floor((target_sz * (1 + padding))*search_size(i));
            current_size = tmp_sz(2)/window_sz(2);
			pos = pos + current_size*cell_size * [vert_delta - 1, horiz_delta - 1];
		end

		%obtain a subwindow for training at newly estimated target position
     	%patch = get_subwindow(im, pos, window_sz);
        target_sz = target_sz * search_size(szid+1);
        tmp_sz = floor((target_sz * (1 + padding)));
        param0 = [pos(2), pos(1), tmp_sz(2)/window_sz(2), 0,...
                    tmp_sz(1)/window_sz(2)/(window_sz(1)/window_sz(2)),0];
        param0 = affparam2mat(param0); 
        patch = uint8(warpimg(double(im), param0, window_sz));
        
        [ss,yy,img] = get_subwindow_no_window(patch,(size(cos_window)*2),size(cos_window));
       %imshow(img);
        S = get_saliency_SCA(img,100);
       % imshow(S);
		xf = fft2(get_features(patch, features, cell_size, cos_window,w2c));
       
        
        kf = conj(xf) .* xf; 
    
        kfn = zeros([size(xf) length(offset)]);
        for j=1:length(offset)
            %obtain a subwindow close to target for regression to 0
            patch = get_subwindow(im, pos+offset(j,:), window_sz);
            xfn = fft2(get_features(patch, features, cell_size, cos_window,w2c));
            kfn(:,:,:,j) = conj(xfn) .*xfn;
        end
 
        %Assumption: Features are independent
      
        num = bsxfun(@times, conj(xf),yf); 
        den = kf + lambda1 + lambda2.*sum(kfn,4);
        ws = num ./ den;
        if isnan(S)
            wf=ws;
        else
            wf = bsxfun(@times,ws,S);
        end
        
		if frame == 1,  %first frame, train with a single image
			model_wf = wf;
		else
			%subsequent frames, interpolate model
			model_wf = (1 - interp_factor) * model_wf + interp_factor * wf;
		end

		%save position and timing
		positions(frame,:) = pos;
		time = time + toc();


		box = [pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])];
        rect_results(frame,:)=box;
   		%visualization
	    if show_visualization,
                stop = update_visualization(frame, box);
			if stop, break, end  %user pressed Esc, stop early
			
			drawnow
% 			pause(0.05)  %uncomment to run slower
		end
		
	end

	if resize_image,
		positions = positions * 2;
        rect_results = rect_results*2;
	end
end