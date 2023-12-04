from PIL import Image
import statistics

name = "DSC03744"

# im = Image.open(f'..\\Lora_experiment\\{name}.png')
im = Image.open(f'D:\二次元\Anime Expo\\{name}.jpg')
pixels = im.load()
dim = 64
new_size = (dim, dim)
out_im = Image.new('RGB', new_size)

kernel_size = im.size[0] // dim
for i in range(dim):
    for j in range(dim):
        r_list = []
        g_list = []
        b_list = []
        for k in range(kernel_size):
            for l in range(kernel_size):
                x = i * kernel_size + k
                y = j * kernel_size + l
                r_list.append(pixels[x, y][0])
                g_list.append(pixels[x, y][1])
                b_list.append(pixels[x, y][2])
        r = statistics.mode(r_list)
        g = statistics.mode(g_list)
        b = statistics.mode(b_list)
        out_im.putpixel((i, j), (r, g, b))

out_im.save(f'..\\Lora_experiment\\{name}_out_s.png')