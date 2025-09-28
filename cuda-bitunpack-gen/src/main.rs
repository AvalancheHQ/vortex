use std::fs::{self, File};
use std::io::Write;
use std::path::Path;
use std::{fmt, io};

use fastlanes::FastLanes;

pub struct IndentedWriter<W: Write> {
    write: W,
    indent: String,
}

impl<W: Write> IndentedWriter<W> {
    fn indent<F>(&mut self, indented: F) -> io::Result<()>
    where
        F: FnOnce(&mut IndentedWriter<W>) -> io::Result<()>,
    {
        let original_ident = self.indent.clone();
        self.indent += "    ";
        let res = indented(self);
        self.indent = original_ident;
        res
    }

    fn write_fmt(&mut self, fmt: fmt::Arguments<'_>) -> io::Result<()> {
        write!(self.write, "{}{}", self.indent, fmt)
    }
}

fn generate_unpack_kernel<T: FastLanes, W: Write>(
    output: &mut IndentedWriter<W>,
    bit_width: usize,
    thread_count: usize,
) -> anyhow::Result<()> {
    let bits = <T>::T;
    let lanes = T::LANES;

    writeln!(
        output,
        "__device__ void unpack_{bit_width}bw_{bits}ow_{thread_count}t(const uint{bits}_t *__restrict a_in_p, uint{bits}_t *__restrict a_out_p) {{"
    )?;

    output.indent(|output| {
        writeln!(output, "int i = threadIdx.x;")?;
        if bit_width == 0 {
            writeln!(
                output,
                "auto out = reinterpret_cast<uint{bits}_t *>(a_out_p);"
            )?;
            writeln!(output, "uint{bits}_t zero = 0ULL;")?;
            writeln!(output)?;
            let per_thread_loop_count = lanes / thread_count;
            for thread_lane in 0..per_thread_loop_count {
                for row in 0..bits {
                    writeln!(output, "out[INDEX({row}, (i * {per_thread_loop_count} + {thread_lane}))] = zero;")?;
                }
            }
        } else if bit_width == bits {
            writeln!(
                output,
                "auto out = reinterpret_cast<uint{bits}_t *>(a_out_p);",
            )?;
            writeln!(
                output,
                "auto in = reinterpret_cast<const uint{bits}_t *>(a_in_p);"
            )?;
            writeln!(output)?;
            let per_thread_loop_count = lanes / thread_count;
            for thread_lane in 0..per_thread_loop_count {
                for row in 0..bits {
                    writeln!(
                        output,
                        "out[INDEX({row}, (i * {per_thread_loop_count} + {thread_lane}))] = in[{lanes} * {row} + (i * {per_thread_loop_count} + {thread_lane})];",
                    )?;
                }
            }
        } else {
            writeln!(
                output,
                "auto out = reinterpret_cast<uint{bits}_t *>(a_out_p);"
            )?;
            writeln!(
                output,
                "auto in = reinterpret_cast<const uint{bits}_t *>(a_in_p);"
            )?;
            writeln!(output, "uint{bits}_t src;")?;
            writeln!(output, "uint{bits}_t tmp;")?;

            let per_thread_loop_count = lanes / thread_count;
            for thread_lane in 0..per_thread_loop_count {
                writeln!(output)?;
                writeln!(output, "src = in[i * {per_thread_loop_count} + {thread_lane}];")?;
                for row in 0..bits {
                    let curr_word = (row * bit_width) / bits;
                    let next_word = ((row + 1) * bit_width) / bits;
                    let shift = (row * bit_width) % bits;

                    if next_word > curr_word {
                        let remaining_bits = ((row + 1) * bit_width) % bits;
                        let current_bits = bit_width - remaining_bits;
                        writeln!(
                            output,
                            "tmp = (src >> {shift}) & MASK(uint{bits}_t, {current_bits});"
                        )?;

                        if next_word < bit_width {
                            writeln!(output, "src = in[i * {per_thread_loop_count} + {thread_lane} + {bits} * {next_word}];")?;
                            writeln!(
                                output,
                                "tmp |= (src << {remaining_bits}) & MASK(uint{bits}_t, {remaining_bits});"
                            )?;
                        }
                    } else {
                        writeln!(
                            output,
                            "tmp = (src >> {shift}) & MASK(uint{bits}_t, {bit_width});"
                        )?;
                    }

                    writeln!(output, "out[INDEX({row}, (i * {per_thread_loop_count} + {thread_lane}))] = tmp;")?;
                }
            }
        }
        Ok(())
    })?;

    writeln!(output, "}}")?;
    Ok(())
}

fn generate_unpack_width_entry_point<T: FastLanes, W: Write>(
    output: &mut IndentedWriter<W>,
    thread_count: usize,
) -> anyhow::Result<()> {
    let bits = <T>::T;

    writeln!(
        output,
        "__device__ void unpack_{bits}bit_{thread_count}t(const uint{bits}_t *__restrict a_in_p, uint{bits}_t *__restrict a_out_p, uint{bits}_t bw) {{"
    )?;
    output.indent(|output| {
        writeln!(output, "switch (bw) {{")?;

        for bw in 0..=bits {
            writeln!(output, "case {bw}:")?;
            output.indent(|output| {
                writeln!(output, "unpack_{bw}bw_{bits}ow_{thread_count}t(a_in_p, a_out_p);",)?;
                writeln!(output, "break;")
            })?;
        }

        writeln!(output, "}}")
    })?;
    writeln!(output, "}}")?;
    Ok(())
}

fn generate_unpack_for_width<T: FastLanes, W: Write>(
    output: &mut IndentedWriter<W>,
    thread_count: usize,
) -> anyhow::Result<()> {
    writeln!(output, "// generated!")?;
    writeln!(output, "#include <cuda.h>")?;
    writeln!(output, "#include <cuda_runtime.h>")?;
    writeln!(output, "#include <stdint.h>")?;
    writeln!(output)?;

    writeln!(output, "namespace fastlanes {{")?;
    writeln!(output, "namespace cuda {{")?;
    writeln!(output)?;

    writeln!(output, "__device__ int FL_ORDER[] = {{0, 4, 2, 6, 1, 5, 3, 7}};")?;
    writeln!(
        output,
        "#define INDEX(row, lane) (FL_ORDER[(row) / 8] * 16 + ((row) % 8) * 128 + (lane))"
    )?;
    writeln!(
        output,
        "#define MASK(T, width) \
            ((width) >= (sizeof(T) * 8) ? \
            (~(T)0) : \
            (((T)1 << ((width) % (sizeof(T) * 8))) - 1))"
    )?;
    writeln!(output)?;

    for bit_width in 0..=<T>::T {
        generate_unpack_kernel::<T, _>(output, bit_width, thread_count)?;
        writeln!(output)?;
    }

    generate_unpack_width_entry_point::<T, _>(output, thread_count)?;

    writeln!(output)?;
    writeln!(output, "}} // namespace cuda")?;
    writeln!(output, "}} // namespace fastlanes")?;

    Ok(())
}

fn generate_unpack<T: FastLanes>(output_dir: &Path, thread_count: usize) -> anyhow::Result<()> {
    let filename = format!("cuda_{}_bit_unpack.cu", T::T);
    let path = output_dir.join(&filename);
    let mut file = File::create(&path)?;
    let mut writer = IndentedWriter {
        write: &mut file,
        indent: "".to_string(),
    };
    generate_unpack_for_width::<T, _>(&mut writer, thread_count)?;
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let output_dir = Path::new("generated");
    fs::create_dir_all(&output_dir)?;

    // Generate for all bit widths and both features
    generate_unpack::<u8>(&output_dir, 32)?;
    generate_unpack::<u16>(&output_dir, 32)?;
    generate_unpack::<u32>(&output_dir, 32)?;
    generate_unpack::<u64>(&output_dir, 16)?;
    println!("\nCUDA bitunpack generation complete!");
    Ok(())
}
