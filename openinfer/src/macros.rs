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
    ($exec:expr, { $($name:ident $( : $ty:ty )?),* $(,)? }) => {
        $( $crate::fetch_executor!(@one $exec, $name $(, $ty)?); )*
    };
    (@one $exec:expr, $name:ident, $ty:ty) => {
        let $name: $ty = $exec.fetch::<$ty>(stringify!($name)).unwrap_or_else(|err| {
            panic!("fetch_executor failed for {}: {}", stringify!($name), err)
        });
    };
    (@one $exec:expr, $name:ident) => {
        let $name = $exec.fetch(stringify!($name)).unwrap_or_else(|err| {
            panic!("fetch_executor failed for {}: {}", stringify!($name), err)
        });
    };
}

#[macro_export]
macro_rules! try_insert_executor {
    ($exec:expr, { $($name:ident : $value:expr),* $(,)? }) => {{
        $( $exec.insert_dynamic(stringify!($name), $value)?; )*
        Ok(())
    }};
}

#[macro_export]
macro_rules! try_fetch_executor {
    ($exec:expr, { $name:ident $( : $ty:ty )? $(,)? }) => {{
        $crate::try_fetch_executor!(@one $exec, $name $(, $ty)?)
    }};
    (@one $exec:expr, $name:ident, $ty:ty) => {{
        $exec.fetch::<$ty>(stringify!($name))
    }};
    (@one $exec:expr, $name:ident) => {{
        $exec.fetch(stringify!($name))
    }};
}
