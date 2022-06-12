function pad_recover_img = FASMF( img  )

        tic;
        img = double( img );
      
        [ M N ] = size( img ) ;
    
        Noise_img = ones( M , N );
        Noise_img( (img > 0 ) & ( img < 255 )) = 0;
       
        winmax = 39  ;
        L = ( winmax - 1 ) / 2;

        padimg = padarray( img , [ L L ] , 'symmetric', 'both' );
        
        NoiseMat = padarray( Noise_img , [ L L ] , 'symmetric', 'both' );
      
        [rows cols] = size( padimg );
        
     
       
        cnt = zeros( M , N ) ;
        cnt = padarray( cnt , [ L L ] , 'symmetric', 'both' ) ;
        
        value = img ; 
        value = padarray( value , [ L L ] , 'symmetric', 'both' ) ;
       
       % val = ones(rows , cols ) ; 
        %Mid_value = zeros( rows , cols , 1e3 ) ; 
        
        
        for j = L+1 : rows-L
            for k = L+1 : cols-L
                if( value(j , k ) == 0 || value( j , k ) == 255 )
                    value( j , k ) = 0  ; 
                else
                  %  Mid_value(j , k , 1) = value( j , k ) ; 
                    cnt( j , k ) = 1 ; 
                end
               
            end
        end
        
        
        
        

        for j = L+1 : rows-L %原圖原點 Y
            for k = L+1 : cols-L   %原圖原點 X      
                if (NoiseMat(j,k) == 1) %是噪點 ?
                    low = -1 ; high = 1 ; cond = 1 ;  
                    while (cond == 1 )
                        p = 1 ;
                      
                        for m=low : high
                            for n=low : high
                                    y(p) = padimg( j + m , k + n );  %把附近的點存入 y 
                                    p = p + 1 ; 
                            end
                        end
                    
                       
                        good = find((y>0) & (y<255)); % 非躁點記下索引
                        [~,c] = size(good);
                     
                          
                        if ( c >= 1 )  %PA1 check c>=2, for PA2 check c>=1 超過兩個非噪點
                           % mask = D{win_idx} ; 
                            for m = low : 1 : high
                                for n = low : 1 : high
                                    if( padimg( j + m , k + n ) == 0 || padimg( j + m , k + n ) == 255 ) 
                                      %  padimg( j + k - win_idx - n  : j + k + win_idx + n , k + n - win_idx - n : j + k + win_idx + n ) 
                                        value( j + m , k + n ) = value( j + m , k + n ) + mean( y(good ) )  ;     %replaced by weighted of good pixels
                                       % Mid_value( j + m , k + n , val( j + m , k + n )  ) = median( y(good) );
                                        cnt(j + m , k + n ) = cnt( j + m , k + n ) + 1 ; 
                                       % val(j + m , k + n ) = val(j + m, k + n ) + 1 ; 
                                    end
                                end
                            end
                            
                            cond = 0 ;
                        else                    %if we did not reach the maximum window size and if we did not find noise-free mean
                            low = low - 1;
                            high = high + 1 ;
                         
                                if (high > L ) 
                        
                                  value( j , k ) = value( j , k ) +  mean( y ); %replace by mean of good pixels 
                                  %Mid_value( j , k , val( j , k ) ) = median( y );
                                  cnt( j , k ) = cnt( j , k ) + 1 ; 
                                  %val( j , k ) = val( j , k ) + 1 ; 
                                  cond = 0 ; 
                                end
                        end
                      
                       
                        clear y; 
                        clear good;
                    end
                end
            end
        end
     
  
        pad_recover_img = uint8 (value(L+1 : rows-L, L+1 : cols-L) ./ cnt(L+1 : rows-L, L+1 : cols-L));
        time = toc; 
        
        
        disp(time) ;
end