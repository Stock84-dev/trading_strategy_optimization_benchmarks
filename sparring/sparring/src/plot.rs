use merovingian::variable;
use merovingian::variable::Variable;
use mouse::error::Result;
use mouse::num::NumExt;
use mouse::prelude::*;
use ordered_float::OrderedFloat;
use plotters::backend::BitMapBackend;
use plotters::prelude::*;
use std::collections::HashMap;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::time::Instant;

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
