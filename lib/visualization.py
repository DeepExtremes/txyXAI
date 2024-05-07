from scipy.ndimage import binary_erosion
import numpy as np
from typing import Union, List, Optional
import numpy.typing as npt
from pathlib import Path
from PIL import ImageFont, ImageDraw, Image
import textwrap

#Matplotlib imports
import matplotlib.pyplot as plt, matplotlib as mpl
from matplotlib.colors import LightSource
from matplotlib import cm
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import to_rgb
import matplotlib.patches as mpatches

def add_mask(img:np.ndarray, mask:np.ndarray, colors:dict={0:None, 1:'r'}, 
             mode:Union[str,float]='border'):
    '''
        Superimpose a `mask` over an image `img` using some blending `mode` and
        specific `colors` for every class in `mask`. It expects:
            - mask: (w,h,1)
            - image: (w,h,3)
    '''
    #Check dimensions and attempt to fix simple problems
    assert img.shape[:2] == mask.shape[:2], f'{img.shape[:2]=} != {mask.shape[:2]=}'
    if len(img.shape) == 2: img= img[...,None] #Add 1 channel at the end
    if img.shape[-1]== 1: img= img[...,[0,0,0]] #Repeat channel 3 times
    if len(mask.shape) == 2: mask= mask[...,None] #Add one channel at the end
    
    assert mask.shape[-1]== 1, f'Mask must have 1 channel in the last dimension ({mask.shape=})'
    assert img.shape[-1]== 3, 'RGB channels must be in the last dimension of the image'
    assert len(img.shape)== 3, f'RGB image can only have 3 dimensions (w,h,c). Image has {len(img.shape)}'
    
    #Create an image only with the colors that we are going to add to the original image
    colors_mask= [np.zeros( (img.shape[0], img.shape[1], 1) ) for _ in range(3)]
    unique_colors= np.unique(mask)
    assert set(unique_colors).issubset( set(colors.keys()) ),\
        f'mask contains {set(unique_colors)=}, but only colors for {set(colors.keys())=} were provided' 
    
    #Fill this image with color
    bg_idx= 0
    for i, c in colors.items():
        if c is None: 
            bg_idx= i
            continue
        color= mcolors.to_rgb(c)
        for chan in range(3):
            curr_mask= mask==i
            if mode == 'border': 
                curr_mask= (curr_mask ^ binary_erosion(curr_mask, np.ones((3,3,1)))).astype(curr_mask.dtype)
            else:
                pass
            colors_mask[chan][curr_mask]= color[chan]
    colors_mask= np.concatenate(colors_mask, axis=-1)
    
    #Combine original image and color image
    if mode == 'border': 
        #colors_mask*= ( (mask!=bg_idx) ^ binary_erosion(mask!=bg_idx, np.ones((3,3,1)))).astype(colors_mask.dtype)
        masked_img= np.where(colors_mask.sum(axis=-1, keepdims=True) > 0., colors_mask, img)
    else:
        assert 0 <= mode <= 1, 'mode must either be `border` or a float between 0 and 1'
        masked_img= img * mode + colors_mask * (1-mode)
    
    return masked_img

def plot_dem(dem: np.ndarray, lon: npt.ArrayLike, lat: npt.ArrayLike, 
             rgb: Optional[np.ndarray] = None,
             elevation_factor: float=5., title: Optional[str]=None):
    ''' Plot DEM in 3D 
        Note: Height is exagerated by elevation_factor times in the plot
    '''
    x, y = np.meshgrid(lon, lat)
    z= dem

    # Set up plot
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    
    ls = LightSource(60, 60)
    if rgb is None:
        # To use a custom hillshading mode, override the built-in shading and pass
        # in the rgb colors of the shaded surface calculated from "shade".
        rgb = ls.shade(z, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
    
    #https://stackoverflow.com/questions/30223161/matplotlib-mplot3d-how-to-increase
    #-the-size-of-an-axis-stretch-in-a-3d-plo
    ax.set_box_aspect(aspect = (1,1,1/30*elevation_factor))
    
    #Plot
    surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=rgb, 
                           linewidth=0, antialiased=False, shade=True, lightsource=ls)
    ax.set_xlabel('lon'); ax.set_ylabel('lat'); ax.set_zlabel('elevation (m)')
    ax.view_init(-140, 60)
    ax.invert_xaxis(); ax.invert_zaxis()
    ax.grid(False)
    if title is not None: ax.set_title(title)

    plt.show()

def normalize(img:np.ndarray,
              center:bool=True, #If True, it makes sure that 0 -> central color of the cmap
              method:str='minmax', #One of ['minmax', 'percentile']
              limits:Optional[tuple]=None, #Set custom limits. If using minmax norm, they
              #are the minimum and maximum values. If using percentile, they are the lowe
              #n% and the top n% for the normalization
             ):
    allowed_methods= ['minmax', 'percentile']
    assert method in allowed_methods, f'{method=} not in {allowed_methods}'

    #Make copy
    img2= img.copy() 
    if method == 'minmax':
        if limits is not None:
            imin, imax= limits
        else:
            valid_idx= ~np.isnan(img) & ~np.isinf(img2)
            imin, imax= np.min(img2[valid_idx]), np.max(img2[valid_idx])
    elif method == 'percentile':
        limits= (2, 98) if limits is None else limits
        imin, imax= np.percentile(img, torch.Tensor(perc).type(img.dtype))
        
    if center:
        imin= min(imin, -imax)
        imax= max(imax, -imin)
    
    #Final normalization
    img2[img2 >= imax]= imax
    img2[img2 <= imin]= imin
    img_normed= (img2 - imin) / (imax - imin)
    return img_normed, imin, imax

def map2cmap(img:np.ndarray, 
             cmap:str='seismic', #If None, asume image has a shape t3xy and just normalize it
             nan_color:Optional[str]='red', #Color nan pixels 
             inf_color:Optional[str]='yellow', #Color inf pixels 
             zero_color:Optional[str]='magenta', #Color for allzero images
             zero_tol:float=1e-5, #Tolerance for detecting zeros
             **normalize_kwargs #See normalize for full parameter description
            ):
    ''' Converts an tcxy image into a color txy3 image using a cmap.
        It also returns some information needed by `plot_colorbar`.
        If cmap is None, asume the image is t3xy and just normalize it into txy3
    '''         
    #Quick checks
    assert len(img.shape) == 4 and (img.shape[1]==1 or img.shape[1]==3),\
        f'Image must have shape (t, [1,3], w, h). Found {img.shape}'
    
    #Detect bad pixels
    if nan_color is not None: nans= np.any(np.isnan(img.transpose(0, 2, 3, 1)), axis=-1)
    if inf_color is not None: infs= np.any(np.isinf(img.transpose(0, 2, 3, 1)), axis=-1)
    if zero_color is not None: t_zeros= np.all(np.isclose(img.transpose(0, 2, 3, 1), 0., atol=zero_tol), axis=(1,2,3))
    
    #Normalize
    img_normed, imin, imax= normalize(img, **normalize_kwargs)
    
    #c -> rgb
    if cmap is None:
        assert len(img_normed.shape) == 4
        img_mapped= img_normed.transpose(0, 2, 3, 1)
    else:
        cmapfn= plt.get_cmap(cmap).copy() if not isinstance(cmap, mpl.colors.Colormap) else cmap.copy()
        #cmapfn.set_bad(nan_color); cmapfn.set_over(inf_color); cmapfn.set_under(inf_color)
        img_mapped= cmapfn(img_normed)[:,0,:,:,:3] #Ignore alpha channel

    #Paint bad pixels
    if nan_color is not None: img_mapped[nans]= np.array(to_rgb(nan_color))[None, None]
    if inf_color is not None: img_mapped[infs]= np.array(to_rgb(inf_color))[None, None]
    if zero_color is not None: img_mapped[t_zeros]= np.array(to_rgb(zero_color))[None, None]
        
    return img_mapped, (imin, imax, cmap), (nan_color is not None and np.any(nans), 
                                            inf_color is not None and np.any(infs), 
                                            zero_color is not None and np.any(t_zeros))

def plot_colorbar(imin, imax, cmap, ax=None, label=None, orientation='horizontal',
                  figsize=(8,0.1), show=True, labelsize=None):
    ''' Plots a colorbar given its limits and a colormap '''
    if ax is None: plt.figure(figsize=figsize)
    norm = mpl.colors.Normalize(vmin=imin, vmax=imax)
    cb1 = mpl.colorbar.ColorbarBase(ax if ax is not None else plt.gcf().gca(), 
                                    cmap=cmap, norm=norm, orientation=orientation)
    if label is not None: cb1.set_label(label)
    if show and ax is None: plt.show()
    if labelsize is not None: cb1.ax.tick_params(labelsize=labelsize)

def percentile_mask(img, perc=[1-0.98, 0.98]):
    p1, p2= torch.quantile(img, torch.Tensor(perc).type(img.dtype))
    img_mask= torch.zeros_like(img).int()
    img_mask[img<p1]= -1
    img_mask[img>=p2]= 1
    return img_mask

def plot_maps(images:list, #Image with shape (t, {1,3}, h, w) 
              cmaps:Optional[list]=None, #List of mpl cmaps. E.g: ['PiYG', 'hot']
              centers:Optional[list]=None, #List of bools defining whether to force the colorbar to be centered
              limits:Optional[list]=None, #List of tuples of two floats, defining the limits of the colorbar
              title:Optional[str]=None, #Title of the plot
              masks:Optional[list]=None, #List of masks with shape (t, 1, h, w) 
              figsize:Union[tuple, float]=1., #Tuple of two ints or float (scale of the auto figsize)
              xlabels:Optional[list]=None, #List of x labels
              ylabels:Optional[list]=None, #List of y labels
              classes:Optional[dict]=None, #Legend of the classes in masks
              colorbar_position='vertical', #One of ['horizontal', 'vertical']
              mask_kwargs:dict={}, #Args to pass to `add_mask`, such as `colors`
              cmap_kwargs:dict=dict(  #Args to pass to `nan_color` and `inf_color`
                  nan_color='cyan', #Color nan pixels 
                  inf_color='orangered', #Color inf pixels 
                  zero_color='magenta'), #Color for allzero images
              upscale_factor:int=1, #Upscale image and mask to reduce border width
              show:bool=True,
              backend:str='matplotlib', #If backend is numpy, create image only through array manipulation
              numpy_backend_kwargs:dict={'size':12, 'color':'black', 
                                         'xstep':1, 'ystep':1, #Plot labels every n steps
                                         'labels':'edges', #Plot either on 'edges' or full 'grid'
                                         'font':'OpenSans_Condensed-Regular.ttf'},
              plot_mask_channel=None, #Plot the mask corresponding to image at position plot_mask_channel independently
              matplotlib_backend_kwargs:dict={'text_size':None},
              transpose:bool=False, #If True, time is set along the y axis (by default, it is along the x-xis)
              stack_every:Union[int,bool]=False, #If True, stack the image vertically every `stack_every` images
              max_text_width:int=50, #Maximum text width (in chars). Breaks text labels if surpassed
             ):
    ''' 
        Images must have shape (t, {1,3}, h, w) 
        If image is already rgb, the parameters associated to that image except `limits` will be ignored
        If `backend == `matplotlib`, returns (fig, axes)
        If `backend == `numpy`, returns 2D image array
    '''
    #Basic check
    assert isinstance(images, list), 'Images must be provided in a list: [(t, {1,3}, h, w), ...]'
    ROWS= len(images)
    
    #Set default values for inputs if they are not provided
    cmaps= ['PiYG']*ROWS if cmaps is None else cmaps
    centers= [False]*ROWS if centers is None else centers
    limits= [None]*ROWS if limits is None else limits
    ylabels= ['']*ROWS if ylabels is None else ylabels
    
    #General checks
    assert len(images) == len(cmaps) == len(centers) == len(limits) == len(ylabels),\
        f'{len(images)=}, {len(cmaps)=}, {len(centers)=}, {len(limits)=}, {len(ylabels)=} are not all equal!'
    assert all([len(i.shape) == 4 for i in images]) and \
           all([np.all(np.array(i.shape)[[0,2,3]] == np.array(images[0].shape)[[0,2,3]]) for i in images]) and \
           all([i.shape[1] == 1 or i.shape[1] == 3 for i in images]),\
        f'Images must all have shape (t, [1,3], h, w). Found: {[i.shape for i in images]}'
    allowed_pos= ['horizontal', 'vertical']
    assert colorbar_position in allowed_pos, f'{colorbar_position} not in {allowed_pos}'
    t, c, h, w= images[0].shape
    assert t>0 and c>0 and h>0 and w>0, f'All dimensions of {images[0].shape=} must be > 0'
    xlabels= ['']*t if xlabels is None else xlabels
    assert len(xlabels) == t, f'{len(xlabels)=} != {t=}'
        
    #Mask checks
    if masks is not None:
        assert len(images) == len(masks), f'{len(images)=} != {len(masks)=}'
        assert all([len(m.shape) == 4 for m in masks]) and \
               all([np.all(np.array(i.shape)[[0,2,3]] == np.array((t, h, w))) for i in masks]) and \
               all([i.shape[1] == 1 for i in masks]),\
            f'Masks must all have shape ({t=}, 1, {h=}, {w=}). Found: {[m.shape for m in masks]}'
        #masks= masks[::-1]
        tm, cm, hm, wm= masks[0].shape
    
    #Try to convert to numpy from pytorch if necessary
    images= [img.detach().cpu().numpy() if not isinstance(img, np.ndarray) else img for img in images]

    #1channel -> rgb using a custom cmap:  t, {1,3}, h, w ->  t, 3, h, w 
    images_rgb= [map2cmap(img, cmap=cmap if img.shape[-3] != 3 else None, center=center, 
                          limits=limit, **cmap_kwargs)
                 for img, cmap, center, limit in zip(images, cmaps, centers, limits)]
    #print([(lo(img), cbar) for img, cbar in images_rgb])
    
    #If there are masks, start their processing
    if masks is not None:
        if 'colors' in mask_kwargs.keys():
            colors= mask_kwargs['colors']
            del mask_kwargs['colors']
        else:
            class_keys= classes.keys() if classes is not None else np.unique(masks)[1:]
            colors= {0:None, **{k: plt.get_cmap('tab20')(i/20) for i,k in enumerate(class_keys)}}
            
        #Add extra mask rgb channel if requested
        if plot_mask_channel is not None:
            mask_kwargs2= {**mask_kwargs}
            mask_kwargs2['mode']= 0.
            # tm, cm, hm, wm -> tm * hm, wm, 1 / 3 -> tm, cm, hm, wm 
            mask_image= add_mask(images[0].transpose(0,2,3,1).reshape(-1, wm, 3)*0, 
                                 masks[plot_mask_channel].transpose(0,2,3,1).reshape(-1, w, 1), 
                                 colors=colors, **mask_kwargs2).reshape(t, h, w, 3).transpose(0,3,1,2)
            images_rgb= images_rgb + [map2cmap(mask_image, cmap=None, center=False, limits=(0,1.), 
                                               nan_color=None, inf_color=None, zero_color=None)]
            masks= masks + [masks[plot_mask_channel]*0]
            ylabels= ylabels + ['Mask']
        
        # tm, cm, hm, wm -> tm, hm, wm, cm -> tm * hm, wm, 1
        if transpose:  # tm, cm, hm, wm -> hm, tm, wm, cm -> tm * hm, wm, 1
            masks_joined= np.concatenate([mask.transpose(2,0,3,1)[::-1,:,::-1].reshape(hm, -1, 1)
                                          for mask in masks], axis=0).transpose(1,0,2)
        else:          # tm, cm, hm, wm -> tm, hm, wm, cm -> tm * hm, wm, 1
            masks_joined= np.concatenate([mask.transpose(0,2,3,1)[:,::-1,::-1].reshape(-1, wm, 1)
                                          for mask in masks], axis=1).transpose(1,0,2)
    
    #Combine images into one
    if transpose:  #t, h, w, 3 - > h,  t * w, 3
        img= np.concatenate([img[:,::-1,::-1].swapaxes(0,1).reshape(h, -1, 3) 
                             for img, _, _ in images_rgb], axis=0).transpose(1,0,2)
    else:          #t, h, w, 3 - > t * h, w, 3
        img= np.concatenate([img[:,::-1,::-1].reshape(-1, w, 3) 
                             for img, _, _ in images_rgb], axis=1).transpose(1,0,2)
        
    #Swap labels if transpose
    if transpose: xlabels, ylabels= ylabels, xlabels
    
    #Add masks if there are some
    if masks is not None:
        if upscale_factor > 1:
            upscale= lambda i: i.repeat(upscale_factor, axis=0).repeat(upscale_factor, axis=1)
            img, masks_joined= upscale(img), upscale(masks_joined)
            h*=upscale_factor; w*=upscale_factor
            
        img= add_mask(img, masks_joined, colors=colors, **mask_kwargs)

                
    #Wrap text: we replace _ by - so that textwrap attempts to break on those first
    #Also, process each line separateline (when there are \n already, textwrap interprets
    #those as characters, which we don't want
    def wrap(text):
        text= str(text).replace('_','--')
        text= '\n'.join([textwrap.fill(line, width=max_text_width)
                           for line in text.splitlines() if line.strip() != ''])
        return text.replace('--','_')

    xlabels= [wrap(text) for text in xlabels]
    ylabels= [wrap(text) for text in ylabels]
        
    #--------------------------------------------------#
    #If backend=='numpy', do not use any matplotlib at all
    #Not all functionality is implemented, only xlabels and ylabels are shown
    if backend=='numpy':
        #Extract kwargs
        fontsize= numpy_backend_kwargs.get('size', 12)
        color= numpy_backend_kwargs.get('color', 'black')
        font_name= numpy_backend_kwargs.get('font','OpenSans_Condensed-Regular.ttf')
        xstep= numpy_backend_kwargs.get('xstep', 1)
        ystep= numpy_backend_kwargs.get('ystep', 1)
        labels_pos= numpy_backend_kwargs.get('labels', 'edge')
        
        #Prepare image
        img= (255*img).clip(0, 255).astype(np.uint8)
        pil_im = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_im)
        try:
            font= ImageFont.truetype(font=font_name, size=fontsize)
        except:
            try:
                font_path= str(Path(__file__).parent.resolve() / font_name)
                font= ImageFont.truetype(font=font_path, size=fontsize)
            except Exception as e:
                print(f'Exception: Could not load {font_name=} nor {font_path=}. ({e})')
                font = ImageFont.load_default()
        
        #Swap step if transpose
        if transpose: xstep, ystep= ystep, xstep
        
        #Draw xlabels and ylabels
        for x, xlabel in list(enumerate(xlabels))[::xstep]:
            x_left= int(x*w) + int(fontsize/1.5)
            for y, ylabel in list(enumerate(ylabels))[::ystep]:
                #Prepare printing position
                y_up, y_down= int((y+1)*h) - int(fontsize*1.5), int(y*h) + int(fontsize/1.5)
                if transpose: y_up, y_down= y_down, y_up
                    
                #Print text
                if x==0 or labels_pos=='grid':
                    draw.multiline_text((x_left, y_up), xlabel, font=font, fill=color)
                if y==0 or labels_pos=='grid':
                    draw.multiline_text((x_left, y_down), ylabel, font=font, fill=color)
            
        #To numpy
        img= np.array(pil_im) #img is h * c, t * w, 3
        
        #Stack?
        if stack_every > 0 and t > stack_every:
            #Make image divisible by stack_every
            stacks= t // stack_every 
            if t % stack_every != 0:
                img= np.concatenate([img, np.zeros( ( h*ROWS, (stack_every - (t % stack_every))*w, 3 ))], axis=1)
                stacks+= 1
                
            #Stack image
            img = np.concatenate([img[:, s*stack_every*w:(s+1)*stack_every*w] for s in range(stacks)], axis=0)
            
        return img
        
    #--------------------------------------------------#
    #Matplotlib backend
    #Update colorbars
    cbar_infos, bad_values= [], []
    for i, (_, cbar_info, bad_value) in enumerate(reversed(images_rgb)): 
        #For indentical colorbars, plot only the first one
        #For None colorbars, do not plot
        if cbar_info not in cbar_infos and cbar_info[-1] is not None:
            cbar_infos.append(cbar_info)
        bad_values.append(bad_value)
    #If true at pos 0: all_zeros, 1: nans, 2: infs
    bad_values= np.any(np.array(bad_values), axis=0)
        
    #Plot images    
    if colorbar_position == 'horizontal':
        nax, nay= 1+len(cbar_infos), 1
        if not isinstance(figsize, tuple):
            figsize= (img.shape[1]/w*figsize, img.shape[0]/h*2.*figsize)
        gridspec_kw={'height_ratios': [figsize[0]] + [figsize[0]*0.03]*len(cbar_infos)}
    
    elif colorbar_position == 'vertical':
        nax, nay= 1, 1+len(cbar_infos)
        if not isinstance(figsize, tuple):
            figsize= (img.shape[1]/w*figsize, img.shape[0]/h*1.*figsize)
        gridspec_kw= {'width_ratios': [figsize[1]] + [figsize[1]*0.007]*len(cbar_infos)}
        
    fig, axes = plt.subplots(nax, nay, figsize=figsize, gridspec_kw=gridspec_kw)
    ax0= axes[0] if hasattr(axes, '__len__') else axes
    ax0.grid(False)
    ax0.imshow(img)
    
    #Add legend to the mask
    if np.any(bad_values):
        patches= [mpatches.Patch(color=cmap_kwargs['nan_color'], label='[NaN]')] if bad_values[0] else []
        patches+= [mpatches.Patch(color=cmap_kwargs['inf_color'], label='[Inf]')] if bad_values[1] else []
        patches+= [mpatches.Patch(color=cmap_kwargs['zero_color'], label='[All zeros]')] if bad_values[2] else []
    else: 
        patches= []
    if classes is not None:
        assert masks is not None, '`classes` provided but `masks` were not' 
        patches+= [mpatches.Patch(color=colors[i], label=classes[i]) 
                   for i in classes.keys() if i in colors.keys() and i in np.unique(masks_joined)] 
    if len(patches):
        ax0.legend(handles=patches, bbox_to_anchor=(1.01, 1.), loc='upper left', borderaxespad=0.)
        if matplotlib_backend_kwargs['text_size'] is not None: 
            plt.setp(ax0.get_legend().get_texts(), fontsize=matplotlib_backend_kwargs['text_size'])
    
    #Set the ticks
    if all(i=='' for i in xlabels): 
        ax0.set_xticks([])
    else:
        ax0.set_xticks(np.arange(img.shape[1]//w)*w + w/2)
        ax0.set_xticklabels(xlabels)
        if matplotlib_backend_kwargs['text_size'] is not None: 
            ax0.tick_params(axis='x', which='major', labelsize=matplotlib_backend_kwargs['text_size'])
        
    if all(i=='' for i in ylabels): 
        ax0.set_yticks([])
    else:
        ax0.set_yticks(np.arange(img.shape[0]//h)*h + h/2)
        ax0.set_yticklabels(ylabels)
        if matplotlib_backend_kwargs['text_size'] is not None: 
            ax0.tick_params(axis='y', which='major', labelsize=matplotlib_backend_kwargs['text_size'])

    if title is not None: 
        ax0.set_title(title)
        if matplotlib_backend_kwargs['text_size'] is not None: 
            ax0.title.set_fontsize(matplotlib_backend_kwargs['text_size'])
    
    #Add colorbars
    for i, cbar_info in enumerate(cbar_infos[::-1]): 
        plot_colorbar(*cbar_info, ax=axes[1+i], orientation=colorbar_position, 
                      labelsize=matplotlib_backend_kwargs['text_size'])
        
    fig.set_tight_layout(True)
    if show: plt.show(fig)
        
    return fig, axes