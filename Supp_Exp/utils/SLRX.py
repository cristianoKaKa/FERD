import torch

def simplified_lrx(hsi_img, block_size=10):
    device = hsi_img.device
    
    n_row, n_col, n_band = hsi_img.shape
    
    rx_img_block = torch.zeros((n_row, n_col), dtype=torch.float32, device=device)
    rx_img_surrounding = torch.zeros((n_row, n_col), dtype=torch.float32, device=device)
    rx_img_full = torch.zeros((n_row, n_col), dtype=torch.float32, device=device)
    
    hsi_data_full = hsi_img.view(-1, n_band).to(device)
    mean_full = hsi_data_full.mean(dim=0)
    cov_matrix_full = torch.cov(hsi_data_full.T)
    inv_cov_matrix_full = torch.linalg.inv(cov_matrix_full + 1e-6 * torch.eye(n_band, device=device))
    
    for row_start in range(0, n_row, block_size):
        for col_start in range(0, n_col, block_size):

            row_end = min(row_start + block_size, n_row)
            col_end = min(col_start + block_size, n_col)
            
            block_data = hsi_img[row_start:row_end, col_start:col_end, :].contiguous().view(-1, n_band).to(device)
            
            mean_block = block_data.mean(dim=0)
            cov_matrix_block = torch.cov(block_data.T)
            inv_cov_matrix_block = torch.linalg.inv(cov_matrix_block + 1e-6 * torch.eye(n_band, device=device))
            
            surrounding_data = []
            for i in range(max(0, row_start - block_size), min(n_row, row_end + block_size), block_size):
                for j in range(max(0, col_start - block_size), min(n_col, col_end + block_size), block_size):
                    if (i != row_start or j != col_start):
                        surrounding_block = hsi_img[i:i + block_size, j:j + block_size, :].contiguous().view(-1, n_band).to(device)
                        surrounding_data.append(surrounding_block)
            
            if surrounding_data:
                surrounding_data = torch.cat(surrounding_data, dim=0)
                mean_surrounding = surrounding_data.mean(dim=0)
                cov_matrix_surrounding = torch.cov(surrounding_data.T)
                inv_cov_matrix_surrounding = torch.linalg.inv(cov_matrix_surrounding + 1e-6 * torch.eye(n_band, device=device))
                
                for i in range(row_start, row_end):
                    for j in range(col_start, col_end):
                        pixel = hsi_img[i, j, :].to(device)
                        diff_surrounding = pixel - mean_surrounding
                        mahalanobis_dist_surrounding = torch.sqrt(torch.dot(diff_surrounding, torch.mv(inv_cov_matrix_surrounding, diff_surrounding)))
                        rx_img_surrounding[i, j] = mahalanobis_dist_surrounding
            
            for i in range(row_start, row_end):
                for j in range(col_start, col_end):
                    pixel = hsi_img[i, j, :]
                    
                    diff_block = pixel - mean_block
                    mahalanobis_dist_block = torch.sqrt(torch.dot(diff_block, torch.mv(inv_cov_matrix_block, diff_block)))
                    rx_img_block[i, j] = mahalanobis_dist_block
                    
                    diff_full = pixel - mean_full
                    mahalanobis_dist_full = torch.sqrt(torch.dot(diff_full, torch.mv(inv_cov_matrix_full, diff_full)))
                    rx_img_full[i, j] = mahalanobis_dist_full
    
    return rx_img_block, rx_img_surrounding, rx_img_full

def To01(X):
    data_min = torch.min(X)
    data_max = torch.max(X)
    X = (X - data_min) / (data_max - data_min)
    return X

def SLRX(X, block_num=10):

    X = X.detach()
    X = To01(X)

    n, a, b, c = X.shape
    block_size = a//block_num
    reshaped = X.squeeze().permute(1,2,0)
    d_matrix_l, d_matrix_r, d_matrix = simplified_lrx(reshaped, block_size=block_size)

    d_matrix_l = To01(d_matrix_l)
    d_matrix_r = To01(d_matrix_r)
    d_matrix = To01(d_matrix)

    return (d_matrix_l*d_matrix_r*d_matrix).cpu().numpy()