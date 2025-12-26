#[macro_export]
macro_rules! insert_executor {
    ($exec:expr, { $($name:ident : $value:expr),* $(,)? }) => {
        $( $exec
            .insert_dynamic(stringify!($name), $value)
            .unwrap_or_else(|err| {
                panic!("insert_executor failed for {}: {}", stringify!($name), err)
            }); )*
    };
}

#[macro_export]
macro_rules! fetch_executor {
    ($exec:expr, { $($name:ident : $ty:ty),* $(,)? }) => {
        $( let $name: ::openinfer::Tensor<$ty> = $exec
            .fetch_typed::<$ty>(stringify!($name))
            .unwrap_or_else(|err| {
                panic!("fetch_executor failed for {}: {}", stringify!($name), err)
            }); )*
    };
}

#[macro_export]
macro_rules! try_insert_executor {
    ($exec:expr, { $($name:ident : $value:expr),* $(,)? }) => {{
        let mut res: Result<(), anyhow::Error> = Ok(());
        $( if res.is_ok() {
            res = $exec.insert_dynamic(stringify!($name), $value);
        } )*
        res
    }};
}

#[macro_export]
macro_rules! try_fetch_executor {
    ($exec:expr, { $name:ident : $ty:ty $(,)? }) => {{
        $exec.fetch_typed::<$ty>(stringify!($name))
    }};
}
