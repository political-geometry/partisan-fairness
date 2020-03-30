background_line_width = 1.0
ups_line_width = 2.0
ups_color = [0.8, 0.2, 0.9]
ups_color_reversed = [0.8, 0.2, 0.9, 0.2]
ups_color_shaded = [0.9, 0.9, 0.9, 0.8]

grid_color = [0.5, 0.5, 0.5, 0.5]

actual_color = [0.2, 0.6, 0.2, 1.0]
scatter_edge_color = [0.2, 0.8, 0.2, 1.0]
bias_color = [1.0, 0.7, 0.0, 0.7]
mean_median_color = [0, 0.7, 0.9, 0.7]
partisan_symmetry_color = [0.97, 0.5, 0.5, 0.8]
symmetric_color = [1.0, 0.2, 0.2, 1.0]
above_line = [0.9, 0.8, 0.8]
below_line = [0.8, 0.8, 0.9]


def plot_settings_to_str(ups, bias, mean_median, partisan_symmetry):
    if ups and not bias and not mean_median and partisan_symmetry:
        return "_partisan_symmetry"
    if ups and not bias and not mean_median and not partisan_symmetry:
        return "_ups"
    if ups and bias and not mean_median:
        return "_ups_bias"
    if ups and not bias and mean_median:
        return "_ups_mean_median"
    if not ups:
        return "_actual_only"

    return "_all"



def combine_as_grid(states, year, plot_settings):
    combined_filenames = []
    for state in states:
        combined_filenames.append('outputs/' +
                                  state + '_' +
                                  str(year) + plot_settings + '.png')

    images = list(map(Image.open, combined_filenames))
    widths, heights = zip(*(i.size for i in images))
    max_width = max(widths)
    max_height = max(heights)

    new_image = Image.new('RGB', (2 * max_width, 2 * max_height))
    draw = ImageDraw.Draw(new_image)
    font = ImageFont.truetype('/Library/Fonts/Arial Unicode.ttf', 240)
    count = 0

    x_buffer = max_width * 0.2
    y_buffer = x_buffer / 4.0
    for im in images:
        x_offset = int((count % 2) * max_width)
        y_offset = int(floor(count / 2) * max_height)

        new_image.paste(im, (x_offset, y_offset))
        draw.text((int(x_offset + x_buffer), y_offset + y_buffer), states[count], font=font, fill='#000000')

        count = count + 1

    new_image.save('outputs/combined' + plot_settings + '.png')



def combine_all():
    states = ["IN", "MN", "MD", "OH"]
    plot_settings = plot_settings_to_str(False, False, False, False)
    combine_as_grid(states, 2016, plot_settings)
    plot_settings = plot_settings_to_str(True, False, False, False)
    combine_as_grid(states, 2016, plot_settings)
    plot_settings = plot_settings_to_str(True, True, True, False)
    combine_as_grid(states, 2016, plot_settings)
    plot_settings = plot_settings_to_str(True, False, False, True)
    combine_as_grid(states, 2016, plot_settings)


