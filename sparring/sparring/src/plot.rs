use merovingian::variable;
use merovingian::variable::Variable;
use mouse::error::Result;
use mouse::num::NumExt;
use mouse::prelude::*;
use ordered_float::OrderedFloat;
use plotters::backend::BitMapBackend;
use plotters::prelude::*;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::time::Instant;

pub fn save_hist(name: &str, hist: &[u32], bucket_size: f32) -> Result<()> {
    let mut out = String::new();
    let values = hist
        .iter()
        .enumerate()
        .map(|x| (x.0 as f32 * bucket_size, *x.1 as f32))
        .for_each(|x| {
            use std::fmt::Write;
            writeln!(&mut out, "{:.1}: {}", x.0, x.1);
        });
    let mut file = File::create(format!("{}_hist.txt", name))?;
    use std::io::Write;
    file.write_all(out.as_bytes())?;
    Ok(())
}

pub fn save_pairs(name: &str, pairs: &mut Vec<crate::context::Pair>) -> Result<()> {
    let mut out = String::new();
    pairs.sort_by_key(|x| OrderedFloat(x.balance));
    let values = pairs
        .iter()
        // .enumerate()
        // .map(|x| (x.0 as f32 * bucket_size, *x.1 as f32))
        .for_each(|x| {
            use std::fmt::Write;
            writeln!(
                &mut out,
                "{} {} {} {}: {:.1} {}",
                x.params[0], x.params[1], x.params[2], x.params[3], x.account.balance, x.account.n_trades,
            );
        });
    let mut file = File::create(format!("{}_pairs.txt", name))?;
    use std::io::Write;
    file.write_all(out.as_bytes())?;
    Ok(())
}

fn plot() -> Result<()> {
    let mut balance = 0.0f32;
    let mut variables = vec![
        Variable {
            min: 60.,
            max: 13440.,
            value: 60.,
            stride: 60.,
        },
        Variable {
            min: 0.0,
            max: 4.0,
            value: 0.0,
            stride: 1.0,
        },
        // there are some between 400 and 500 but not between 500 and 1000
        Variable {
            min: 2.,
            max: 482.,
            value: 2.,
            stride: 1.,
        },
        Variable {
            min: 0.,
            max: 100.,
            value: 0.,
            stride: 1.,
        },
        Variable {
            min: 0.,
            max: 100.,
            value: 0.,
            stride: 1.,
        },
    ];
    let var_id = 0;

    #[derive(Debug)]
    struct Value {
        min: f32,
        max: f32,
        average: f32,
        count: f32,
        stddev: f32,
    }
    let mut values: HashMap<OrderedFloat<f32>, Value> = HashMap::default();
    let mut file = BufReader::with_capacity(8192 * 16, std::fs::File::open("balances.bin")?);
    let mut now = Instant::now();
    loop {
        if file
            .read_exact(unsafe { balance.as_u8_slice_mut() })
            .is_err()
        {
            break;
        }
        //        if now.elapsed().as_secs() > 10 {
        //            break;
        //        }
        variable::increase(&mut variables, 1);
        match values.get_mut(&OrderedFloat(variables[var_id].value)) {
            None => {
                values.insert(
                    OrderedFloat(variables[var_id].value),
                    Value {
                        min: balance,
                        max: balance,
                        average: balance,
                        count: 1.,
                        stddev: 0.,
                    },
                );
            }
            Some(value) => {
                //                value.min.min_mut(balance);
                value.max.max_mut(balance);
                value.average += balance;
                value.count += 1.;
            }
        }
    }
    println!("stage complete");
    for (x, value) in &mut values {
        value.average /= value.count;
    }
    file.seek(SeekFrom::Start(0))?;
    variable::reset(&mut variables);
    let mut now = Instant::now();
    loop {
        //        if now.elapsed().as_secs() > 5 {
        //            break;
        //        }
        if file
            .read_exact(unsafe { balance.as_u8_slice_mut() })
            .is_err()
        {
            break;
        }
        variable::increase(&mut variables, 1);
        let mut value = values
            .get_mut(&OrderedFloat(variables[var_id].value))
            .unwrap();
        value.stddev += (balance - value.average).powi(2);
    }
    println!("plotting");
    for item in &mut values {
        item.1.stddev = (item.1.stddev / item.1.count).sqrt();
    }
    let mut series: Vec<_> = values.iter().sorted_by_key(|x| x.0).collect();
    let mut max_rel = values
        .iter()
        .map(|x| OrderedFloat(x.1.stddev / x.1.average))
        .max()
        .unwrap()
        .0;
    let mut min_rel = values
        .iter()
        .map(|x| OrderedFloat(x.1.stddev / x.1.average))
        .min()
        .unwrap()
        .0;
    let mut max = values
        .iter()
        .map(|x| OrderedFloat(x.1.average))
        .max()
        .unwrap()
        .0;
    let mut min = values
        .iter()
        .map(|x| OrderedFloat(x.1.average))
        .min()
        .unwrap()
        .0;
    let path = format!("{}.png", var_id);
    let root = BitMapBackend::new(&path, (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        //        .caption("y=x^2", ("sans-serif", 50).into_font())
        .margin(5u32)
        .x_label_area_size(30u32)
        .y_label_area_size(40u32)
        .right_y_label_area_size(40u32)
        .build_cartesian_2d(variables[var_id].min..variables[var_id].max, min..max)?
        .set_secondary_coord(
            variables[var_id].min..variables[var_id].max,
            min_rel..max_rel,
        );
    let y1_desc = "average";
    let y2_desc = "stddev / average";

    chart.configure_mesh().y_desc(y1_desc).draw()?;
    chart.configure_secondary_axes().y_desc(y2_desc).draw()?;

    chart
        .draw_secondary_series(LineSeries::new(
            series.iter().map(|x| (x.0 .0, x.1.stddev / x.1.average)),
            &BLUE,
        ))?
        .label(y2_desc)
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));
    chart
        .draw_series(LineSeries::new(
            series.iter().map(|x| (x.0 .0, x.1.average)),
            &RED,
        ))?
        .label(y1_desc)
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;
    dbg!(values);

    Ok(())
}

pub fn histogram(name: &str, data: &[u32]) -> Result<()> {
    dbg!(data);
    let path = format!("{} histogram.png", name);
    let root = BitMapBackend::new(&path, (1680 / 2, 1050 / 2)).into_drawing_area();

    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .x_label_area_size(35)
        .y_label_area_size(40)
        .margin(5)
        .caption("Histogram Test", ("sans-serif", 50.0))
        .build_cartesian_2d((0u32..10u32).into_segmented(), 0u32..10u32)?;

    chart
        .configure_mesh()
        .disable_x_mesh()
        .bold_line_style(&WHITE.mix(0.3))
        .y_desc("Count")
        .x_desc("Bucket")
        .axis_desc_style(("sans-serif", 15))
        .draw()?;

    chart.draw_series(
        Histogram::vertical(&chart)
            .style(RED.mix(0.5).filled())
            .data(data.iter().map(|x: &u32| (*x, 1))),
    )?;

    // To avoid the IO failure being ignored silently, we manually call the present function
    root.present().expect("Unable to write result to file, please make sure 'plotters-doc-data' dir exists under current dir");

    Ok(())
}

pub fn plot_xy(name: &str, values: &[(f32, f32)]) -> Result<()> {
    let path = format!("{}.png", name);
    let root = BitMapBackend::new(&path, (1680 / 2, 1050 / 2)).into_drawing_area();
    root.fill(&WHITE)?;
    // find min and max values
    let mut min = values[0].1;
    let mut max = values[0].1;
    for value in values {
        min.min_mut(value.1);
        max.max_mut(value.1);
    }
    let mut chart = ChartBuilder::on(&root)
        //        .caption("y=x^2", ("sans-serif", 50).into_font())
        .margin(5u32)
        .x_label_area_size(30u32)
        .y_label_area_size(40u32)
        .build_cartesian_2d(values[0].0..values.last().unwrap().0, min..max)?;

    chart.configure_mesh().y_desc(name).draw()?;
    chart
        .draw_series(LineSeries::new(values.iter().cloned(), &RED))?
        .label(name)
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;
    Ok(())
}

pub fn plot_values(name: &str, values: &[f32]) -> Result<()> {
    let path = format!("{}.png", name);
    let root = BitMapBackend::new(&path, (1680 / 2, 1050 / 2)).into_drawing_area();
    root.fill(&WHITE)?;
    // find min and max values
    let mut min = values[0];
    let mut max = values[0];
    for value in values {
        min.min_mut(*value);
        max.max_mut(*value);
    }
    let mut chart = ChartBuilder::on(&root)
        //        .caption("y=x^2", ("sans-serif", 50).into_font())
        .margin(5u32)
        .x_label_area_size(30u32)
        .y_label_area_size(40u32)
        .build_cartesian_2d(0.0f32..values.len() as f32, min..max)?;

    chart.configure_mesh().y_desc(name).draw()?;
    chart
        .draw_series(LineSeries::new(
            values.iter().enumerate().map(|x| (x.0 as f32, *x.1)),
            &RED,
        ))?
        .label(name)
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;
    Ok(())
}
